targetScope = 'resourceGroup'

@description('Must remain false in source verification.')
param provisionWatchdog bool = false
param location string = resourceGroup().location
@minLength(3)
@maxLength(10)
param suffix string
param subscriptionId string
param pilotResourceGroupName string
@minLength(3)
@maxLength(16)
param pilotSuffix string
param controllerPrincipalObjectId string
@description('Immutable UTC cleanup deadline supplied by the future authorized controller.')
param deadlineUtc string

var managementEndpoint = environment().resourceManager

resource ledger 'Microsoft.Storage/storageAccounts@2023-05-01' = if (provisionWatchdog) {
  name: 'kovaledger${suffix}'
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    minimumTlsVersion: 'TLS1_2'
    publicNetworkAccess: 'Enabled'
  }
}

resource ledgerBlob 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = if (provisionWatchdog) {
  parent: ledger
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource lifecycleContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = if (provisionWatchdog) {
  parent: ledgerBlob
  name: 'lifecycle-ledger'
  properties: {
    publicAccess: 'None'
  }
}

resource controllerLedgerWriter 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (provisionWatchdog) {
  name: guid(ledger!.id, controllerPrincipalObjectId, 'ledger-writer')
  scope: ledger
  properties: {
    principalId: controllerPrincipalObjectId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  }
}

resource pilotGroup 'Microsoft.Resources/resourceGroups@2022-09-01' existing = {
  name: pilotResourceGroupName
  scope: subscription(subscriptionId)
}

resource workflow 'Microsoft.Logic/workflows@2019-05-01' = if (provisionWatchdog) {
  name: 'kova-pilot-watchdog-${suffix}'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    state: 'Enabled'
    parameters: {
      deadlineUtc: {
        value: deadlineUtc
      }
    }
    definition: {
      '$schema': 'https://schema.management.azure.com/schemas/2016-06-01/Microsoft.Logic.json#'
      contentVersion: '1.0.0.0'
      parameters: {
        deadlineUtc: {
          type: 'String'
        }
      }
      triggers: {
        every_minute: {
          type: 'Recurrence'
          recurrence: {
            frequency: 'Minute'
            interval: 1
          }
          conditions: [
            '@greaterOrEquals(ticks(utcNow()), ticks(parameters(\'deadlineUtc\')))'
          ]
        }
      }
      actions: {
        deallocate_after_deadline: {
          type: 'Http'
          inputs: {
            method: 'POST'
            uri: '${managementEndpoint}subscriptions/${subscriptionId}/resourceGroups/${pilotResourceGroupName}/providers/Microsoft.Compute/virtualMachines/kova-t4-${pilotSuffix}/deallocate?api-version=2024-07-01'
            authentication: {
              type: 'ManagedServiceIdentity'
              audience: managementEndpoint
            }
          }
          runAfter: {}
        }
        delete_pilot_group: {
          type: 'Http'
          inputs: {
            method: 'DELETE'
            uri: '${managementEndpoint}subscriptions/${subscriptionId}/resourceGroups/${pilotResourceGroupName}?api-version=2022-09-01'
            authentication: {
              type: 'ManagedServiceIdentity'
              audience: managementEndpoint
            }
          }
          runAfter: {
            deallocate_after_deadline: [
              'Succeeded'
              'Failed'
              'TimedOut'
            ]
          }
        }
      }
      outputs: {}
    }
  }
}

module watchdogPilotContributor 'three-family-watchdog-pilot-role.bicep' = if (provisionWatchdog) {
  name: 'watchdog-pilot-role-${suffix}'
  scope: pilotGroup
  params: {
    watchdogPrincipalId: workflow!.identity.principalId
  }
}

output resourceCreationAuthorized bool = false
output spendingAuthorized bool = false
output ledgerAccountName string = provisionWatchdog ? ledger!.name : ''
