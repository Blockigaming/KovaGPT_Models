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

// A private VM needs a predictable outbound route for the signed driver,
// dependencies and pinned model snapshot. The IP belongs only to the outbound
// NAT gateway; the VM NIC never has an inbound public IP.
resource egressIp 'Microsoft.Network/publicIPAddresses@2024-05-01' = if (provisionPilot) {
  name: 'kova-t4-egress-ip-${suffix}'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

resource egressNat 'Microsoft.Network/natGateways@2024-05-01' = if (provisionPilot) {
  name: 'kova-t4-egress-nat-${suffix}'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    idleTimeoutInMinutes: 4
    publicIpAddresses: [
      { id: egressIp!.id }
    ]
  }
}

resource egressRules 'Microsoft.Network/networkSecurityGroups@2024-05-01' = if (provisionPilot) {
  name: 'kova-t4-egress-nsg-${suffix}'
  location: location
  properties: {
    securityRules: [
      {
        name: 'allow-https-egress'
        properties: {
          priority: 100
          direction: 'Outbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
      {
        name: 'allow-http-package-egress'
        properties: {
          priority: 110
          direction: 'Outbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '80'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
      {
        name: 'deny-other-egress'
        properties: {
          priority: 120
          direction: 'Outbound'
          access: 'Deny'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Deny'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

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
          defaultOutboundAccess: false
          natGateway: { id: egressNat!.id }
          networkSecurityGroup: { id: egressRules!.id }
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
output vmPublicIpCreated bool = false
output deploymentAuthorized bool = false
