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
                def mensaje = """
✅ Pipeline MLOps finalizado OK

📌 Build: ${env.BUILD_NUMBER}
📌 Job: ${env.JOB_NAME}
⏱ Duración: ${currentBuild.durationString}
"""
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
                def mensaje = """
❌ Pipeline MLOps falló

📌 Build: ${env.BUILD_NUMBER}
📌 Job: ${env.JOB_NAME}
❗ Error: ${currentBuild.currentResult}

🔍 Revisar logs: ${env.BUILD_URL}console
"""
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

