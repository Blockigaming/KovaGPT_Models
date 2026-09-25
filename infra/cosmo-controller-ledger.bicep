targetScope = 'resourceGroup'

@description('Source preparation only. Creation and the subsequent policy lock require separate owner approval.')
param provisionLedger bool = false
@description('Use a dedicated ledger group, outside both pilot and watchdog cleanup groups.')
param pilotResourceGroupId string
param watchdogResourceGroupId string
@description('Independent controller service principal; never the training VM or watchdog identity.')
@minLength(36)
@maxLength(36)
param controllerPrincipalId string

// Keep the selected SKU, region and retention aligned with the reviewed price component.
var isolated = toLower(resourceGroup().id) != toLower(pilotResourceGroupId) && toLower(resourceGroup().id) != toLower(watchdogResourceGroupId)
var createLedger = provisionLedger && isolated
var storageName = 'kovaledger${uniqueString(subscription().subscriptionId, resourceGroup().id)}'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = if (createLedger) {
  name: storageName
  location: 'eastus'
  kind: 'BlobStorage'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
    // Public HTTPS endpoint, private authenticated data; no private endpoint/NAT charge.
    publicNetworkAccess: 'Enabled'
  }
}

resource service 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = if (createLedger) {
  parent: storage
  name: 'default'
  properties: {
    isVersioningEnabled: false
    deleteRetentionPolicy: { enabled: false }
    containerDeleteRetentionPolicy: { enabled: false }
  }
}

resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = if (createLedger) {
  parent: service
  name: 'cosmo-ledger'
  properties: {
    publicAccess: 'None'
  }
}

// ARM creates an UNLOCKED policy. Only a separate ETag-conditional lock operation
// makes this acceptable to ControllerLedger._policy(); deployment alone is not ready.
resource retention 'Microsoft.Storage/storageAccounts/blobServices/containers/immutabilityPolicies@2023-05-01' = if (createLedger) {
  parent: container
  name: 'default'
  properties: {
    immutabilityPeriodSinceCreationInDays: 30
    allowProtectedAppendWrites: true
    allowProtectedAppendWritesAll: false
  }
}

resource controllerData 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (createLedger) {
  name: guid(container.id, controllerPrincipalId, 'ledger-data')
  scope: container
  properties: {
    principalId: controllerPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  }
}

resource controllerPolicyReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (createLedger) {
  name: guid(container.id, controllerPrincipalId, 'ledger-policy-reader')
  scope: container
  properties: {
    principalId: controllerPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'acdd72a7-3385-48ef-bd42-f606fba81ae7')
  }
}

output cleanupGroupsIsolated bool = isolated
output storageAccountName string = createLedger ? storage.name : ''
output containerName string = createLedger ? container.name : ''
output retentionDays int = 30
output policyLockRequired bool = true
output ledgerInitialized bool = false
output resourceCreationAuthorized bool = false
output deploymentAuthorized bool = false
