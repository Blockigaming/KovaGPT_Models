targetScope = 'resourceGroup'

@description('A deployment controller must separately authorize resource creation. Compilation grants nothing.')
param provisionStaging bool = false
@description('Separate gate for app resources; never enables Kova model execution inside the image.')
param activateModelApps bool = false
@minLength(3)
@maxLength(12)
param stagingSuffix string
param location string
param infrastructureSubnetId string
param imagePullIdentityId string
param registryServer string
param coreImage string
param ultraImage string
@allowed([
  'Consumption-GPU-NC24-A100'
  'Consumption-GPU-NC8as-T4'
])
param gpuProfileType string
@minValue(1)
@maxValue(2)
param maxReplicas int

var engines = ['core', 'ultra']

resource modelEnvironment 'Microsoft.App/managedEnvironments@2025-07-01' = if (provisionStaging) {
  name: 'kova-model-staging-${stagingSuffix}'
  location: location
  tags: {
    product: 'KovaGPT'
    purpose: 'model-staging-only'
  }
  properties: {
    publicNetworkAccess: 'Disabled'
    appLogsConfiguration: {
      destination: 'none'
    }
    vnetConfiguration: {
      internal: true
      infrastructureSubnetId: infrastructureSubnetId
    }
    peerTrafficConfiguration: {
      encryption: {
        enabled: true
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
      {
        name: 'model-gpu'
        workloadProfileType: gpuProfileType
      }
    ]
  }
}

resource modelApps 'Microsoft.App/containerApps@2025-07-01' = [for engine in engines: if (provisionStaging && activateModelApps) {
  name: 'kova-${engine}-staging-${stagingSuffix}'
  location: location
  tags: {
    product: 'KovaGPT'
    purpose: 'model-staging-only'
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${imagePullIdentityId}': {}
    }
  }
  properties: {
    environmentId: modelEnvironment!.id
    workloadProfileName: 'model-gpu'
    configuration: {
      activeRevisionsMode: 'Single'
      registries: [
        {
          server: registryServer
          identity: imagePullIdentityId
        }
      ]
      ingress: {
        external: false
        allowInsecure: false
        targetPort: 8080
        transport: 'http'
      }
    }
    template: {
      terminationGracePeriodSeconds: 60
      containers: [
        {
          name: 'model-service'
          image: engine == 'core' ? coreImage : ultraImage
          resources: {
            cpu: gpuProfileType == 'Consumption-GPU-NC24-A100' ? 24 : 8
            memory: gpuProfileType == 'Consumption-GPU-NC24-A100' ? '220Gi' : '56Gi'
          }
          env: [
            { name: 'KOVA_MODEL_EXECUTION_ENABLED', value: 'false' }
            { name: 'HF_HUB_OFFLINE', value: '1' }
            { name: 'TRANSFORMERS_OFFLINE', value: '1' }
            { name: 'DO_NOT_TRACK', value: '1' }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: { path: '/readyz', port: 8080 }
              initialDelaySeconds: 5
              periodSeconds: 5
              timeoutSeconds: 2
              failureThreshold: 3
            }
            {
              type: 'Liveness'
              httpGet: { path: '/healthz', port: 8080 }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 2
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'bounded-http'
            http: {
              metadata: {
                concurrentRequests: '1'
              }
            }
          }
        ]
      }
    }
  }
}]

output phaseBReady bool = false
output modelExecutionEnabled bool = false
