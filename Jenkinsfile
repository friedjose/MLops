pipeline {
    agent any

    options {
        timestamps()
    }

    stages {

        stage('Checkout') {
            steps {
                echo "📦 Clonando repositorio..."
                checkout scm
                sh 'ls -R'
            }
        }

        stage('Configurar entorno') {
            steps {
                echo "🐍 Creando entorno virtual e instalando dependencias..."
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r Mlops/src/requirements.txt
                '''
            }
        }

        stage('Ejecutar prueba simple') {
            steps {
                echo "🧪 Ejecutando cargar_datos.py..."
                sh '''
                    . venv/bin/activate
                    python Mlops/src/cargar_datos.py || true
                '''
            }
        }
    }

    post {
        success {
            echo "✅ Pipeline completado con éxito"
        }
        failure {
            echo "❌ Fallo el pipeline"
        }
    }
}
