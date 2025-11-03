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
                    
                    // Verificar si el venv ya existe (para acelerar builds)
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
                    
                    // Crear venv si no existe
                    sh '''
                        if [ ! -d "venv" ]; then
                            python3 -m venv venv
                        fi
                        
                        # Activar y actualizar pip
                        . venv/bin/activate
                        pip install --upgrade pip --quiet
                        
                        # Instalar dependencias
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

        stage('Smoke test: cargar_datos.py') {
            steps {
                echo "🧪 Probando carga de datos..."
                sh '''
                    . venv/bin/activate
                    python3 Mlops/src/cargar_datos.py
                '''
            }
        }
    }

    post {
        success {
            echo "✅ Pipeline completado con éxito"
            emailext (
                subject: "✅ ÉXITO | Pipeline MLOps finalizado",
                body: """
Hola equipo 👋,

El pipeline MLOps terminó correctamente ✅

📌 Repositorio: ${env.GIT_URL}
📌 Rama: ${env.GIT_BRANCH}
📌 Job: ${env.JOB_NAME}
📌 Build: ${env.BUILD_NUMBER}
⏱ Duración: ${currentBuild.durationString}
✅ Estado: SUCCESS

Saludos,  
Jenkins MLOps 🤖
""",
                to: "tu_correo@ejemplo.com"
            )
        }
        failure {
            echo "❌ Falló el pipeline"
            emailext (
                subject: "❌ ERROR | Pipeline MLOps falló",
                body: """
Hola equipo ⚠️,

El pipeline MLOps falló ❌

📌 Repositorio: ${env.GIT_URL}
📌 Rama: ${env.GIT_BRANCH}
📌 Job: ${env.JOB_NAME}
📌 Build: ${env.BUILD_NUMBER}
⏱ Duración: ${currentBuild.durationString}
❗ Error: ${currentBuild.currentResult}

Por favor revisar logs aquí:
${env.BUILD_URL}console

Saludos,  
Jenkins MLOps 🤖
""",
                to: "josefervi50000@gmail.com"
            )
        }
        cleanup {
            echo "🧹 Limpiando workspace (opcional)..."
            // Descomentar si quieres limpiar después de cada build
            // cleanWs()
        }
    }

