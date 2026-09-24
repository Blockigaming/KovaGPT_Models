targetScope = 'resourceGroup'

@description('Must remain false in source verification; a signed external controller supplies true only after paid authorization.')
param provisionPilot bool = false
param location string = resourceGroup().location
@minLength(3)
@maxLength(16)
param suffix string
param adminUsername string
@description('Exact immutable Marketplace image version from a reviewed East US image listing. No default or latest alias.')
@minLength(8)
@allowed(['24.04.202609040'])
param ubuntuImageVersion string
@secure()
param sshPublicKey string

var vmName = 'kova-t4-${suffix}'

resource vnet 'Microsoft.Network/virtualNetworks@2024-05-01' = if (provisionPilot) {
  name: 'kova-t4-vnet-${suffix}'
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: ['10.91.0.0/16']
    }
    subnets: [
      {
        name: 'pilot'
        properties: {
          addressPrefix: '10.91.1.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
    ]
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2024-05-01' = if (provisionPilot) {
  name: 'kova-t4-nic-${suffix}'
  location: location
  properties: {
    ipConfigurations: [{
      name: 'private'
      properties: {
        privateIPAllocationMethod: 'Dynamic'
        subnet: { id: vnet!.properties.subnets[0].id }
      }
    }]
  }
}

resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = if (provisionPilot) {
  name: vmName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  tags: {
    product: 'KovaGPT'
    purpose: 'bounded-three-family-pilot'
    production: 'false'
  }
  properties: {
    hardwareProfile: {
      vmSize: 'Standard_NC4as_T4_v3'
    }
    osProfile: {
      computerName: vmName
      adminUsername: adminUsername
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: 'ubuntu-24_04-lts'
        sku: 'server'
        version: ubuntuImageVersion
      }
      osDisk: {
        createOption: 'FromImage'
        deleteOption: 'Delete'
        managedDisk: {
          storageAccountType: 'StandardSSD_LRS'
        }
        diskSizeGB: 64
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic!.id
          properties: {
            deleteOption: 'Delete'
            primary: true
          }
        }
      ]
    }
  }
}

resource nvidiaGpuDriver 'Microsoft.Compute/virtualMachines/extensions@2024-03-01' = if (provisionPilot) {
  parent: vm
  name: 'NvidiaGpuDriverLinux'
  location: location
  properties: {
    publisher: 'Microsoft.HpcCompute'
    type: 'NvidiaGpuDriverLinux'
    typeHandlerVersion: '1.10'
    autoUpgradeMinorVersion: false
    enableAutomaticUpgrade: false
  }
}

output resourceCreationAuthorized bool = false
output publicIpCreated bool = false
output deploymentAuthorized bool = false
