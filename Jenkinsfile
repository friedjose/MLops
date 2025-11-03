pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {
        stage('Checkout') {
            steps {
                echo "📦 Clonando repositorio..."
                checkout scm
            }
        }

        stage('Configurar entorno Python') {
            steps {
                script {
                    echo "🐍 Configurando entorno virtual..."
                    
                    def venvExists = fileExists('venv/bin/activate')
                    
                    if (!venvExists) {
                        echo "⚙️ Creando entorno virtual nuevo..."
                        retry(3) {
                            sh '''
                                apt-get update -qq
                                apt-get install -y python3 python3-venv python3-pip
                            '''
                        }
                    } else {
                        echo "✅ Reutilizando entorno virtual existente"
                    }
                    
                    sh '''
                        if [ ! -d "venv" ]; then
                            python3 -m venv venv
                        fi
                        
                        . venv/bin/activate
                        pip install --upgrade pip --quiet
                        echo "📚 Instalando dependencias de Python..."
                        pip install -r Mlops/requirements.txt --quiet
                    '''
                }
            }
        }

        stage('Pre-checks (pyops)') {
            steps {
                echo "🔍 Verificando estructura del proyecto..."
                sh '''
                    . venv/bin/activate
                    python3 pyops/check_structure.py
                '''

                echo "🛡️ Verificando secretos..."
                sh '''
                    . venv/bin/activate
                    python3 pyops/check_secrets.py
                '''
            }
        }
    }

    post {
        success {
            script {
                def mensaje = "**✅ Pipeline MLOps finalizado exitosamente**\\n\\n" +
                              "**Build:** ${env.BUILD_NUMBER}\\n" +
                              "**Job:** ${env.JOB_NAME}\\n" +
                              "**Duración:** ${currentBuild.durationString}\\n\\n" +
                              "Todos los pasos se ejecutaron correctamente. El entorno fue configurado, las dependencias instaladas y las validaciones superadas.\\n\\n" +
                              "_Este mensaje fue generado automáticamente por Jenkins._"
                httpRequest(
                    httpMode: 'POST',
                    url: 'https://discord.com/api/webhooks/1435014869467533322/752Mi4kROZEL5483Os85_2GEAGktQ7Clzi-ywCcRw5O3JiVcvYfBKH2H8Lz4BVF0ZCye',
                    contentType: 'APPLICATION_JSON',
                    requestBody: """{ "content": "${mensaje}" }"""
                )
            }
        }
        failure {
            script {
                def mensaje = "**❌ Pipeline MLOps falló durante la ejecución**\\n\\n" +
                              "**Build:** ${env.BUILD_NUMBER}\\n" +
                              "**Job:** ${env.JOB_NAME}\\n" +
                              "**Estado:** ${currentBuild.currentResult}\\n" +
                              "**Logs:** ${env.BUILD_URL}console\\n\\n" +
                              "Se recomienda revisar los registros para identificar el origen del error.\\n\\n" +
                              "_Este mensaje fue generado automáticamente por Jenkins._"
                httpRequest(
                    httpMode: 'POST',
                    url: 'https://discord.com/api/webhooks/1435014869467533322/752Mi4kROZEL5483Os85_2GEAGktQ7Clzi-ywCcRw5O3JiVcvYfBKH2H8Lz4BVF0ZCye',
                    contentType: 'APPLICATION_JSON',
                    requestBody: """{ "content": "${mensaje}" }"""
                )
            }
        }
    }
}

