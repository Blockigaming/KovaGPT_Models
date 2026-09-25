targetScope = 'resourceGroup'

@description('Must remain false until the owner approves the exact storage account and retention charges.')
param provisionEvidence bool = false
@minLength(3)
@maxLength(10)
param suffix string
param location string = resourceGroup().location

// Must be deployed to a dedicated group outside the pilot and watchdog groups.
// The independent controller alone receives scoped Blob data permissions later.
resource account 'Microsoft.Storage/storageAccounts@2025-01-01' = if (provisionEvidence) {
  name: 'kova42${suffix}'
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Enabled'
    isHnsEnabled: false
    networkAcls: {
      bypass: 'None'
      defaultAction: 'Allow'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2025-01-01' = if (provisionEvidence) {
  parent: account
  name: 'default'
  properties: {}
}

resource ledger 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = if (provisionEvidence) {
  parent: blobService
  name: 'cosmo-ledger'
  properties: {
    publicAccess: 'None'
  }
}

resource adapters 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = if (provisionEvidence) {
  parent: blobService
  name: 'cosmo-adapters'
  properties: {
    publicAccess: 'None'
  }
}

// Policy creation is unlocked by Azure. The separate, explicitly approved
// ETag-bound lock operation must complete before ControllerLedger.initialize().
resource ledgerPolicy 'Microsoft.Storage/storageAccounts/blobServices/containers/immutabilityPolicies@2025-01-01' = if (provisionEvidence) {
  parent: ledger
  name: 'default'
  properties: {
    immutabilityPeriodSinceCreationInDays: 30
    allowProtectedAppendWrites: true
    allowProtectedAppendWritesAll: false
  }
}

resource adapterPolicy 'Microsoft.Storage/storageAccounts/blobServices/containers/immutabilityPolicies@2025-01-01' = if (provisionEvidence) {
  parent: adapters
  name: 'default'
  properties: {
    immutabilityPeriodSinceCreationInDays: 30
    allowProtectedAppendWrites: false
    allowProtectedAppendWritesAll: false
  }
}

output accountName string = provisionEvidence ? account!.name : ''
output resourceCreationAuthorized bool = false
