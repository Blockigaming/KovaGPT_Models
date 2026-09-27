targetScope = 'resourceGroup'

param watchdogPrincipalId string

resource watchdogPilotContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, watchdogPrincipalId, 'watchdog-pilot-contributor')
  properties: {
    principalId: watchdogPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c')
  }
}
