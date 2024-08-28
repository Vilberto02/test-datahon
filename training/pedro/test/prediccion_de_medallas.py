import pickle
import sys

# 1. Cargar el modelo
def cargar_modelo(ruta_modelo):
    with open(ruta_modelo, 'rb') as file:
        modelo = pickle.load(file)
    return modelo

def interfaz_usuario(modelo):
    print("\n--- Bienvenido al Sistema de Predicción de Medallas ---")
    print("Este sistema le permite predecir el total de medallas que podría ganar un país en los Juegos Olímpicos.")
    print("Solo necesita ingresar el número de medallas de oro, plata y bronce que espera ganar.\n")
    
    while True:
        try:
            # Solicitar la entrada del usuario
            print("Por favor, ingrese el número de medallas de oro, plata y bronce, separadas por comas.")
            print("Ejemplo: si espera ganar 1 medalla de oro, 4 de plata y 3 de bronce, ingrese: 1,4,3")
            entrada = input("Ingrese las medallas (oro, plata, bronce): ")
            
            # Convertir la entrada en números
            oro, plata, bronce = map(int, entrada.split(','))
            
            # Hacer la predicción usando el modelo
            prediccion = modelo.predict([[oro, plata, bronce]])
            
            # Mostrar el resultado de manera clara
            print(f"\nPredicción: Con {oro} medalla(s) de oro, {plata} medalla(s) de plata y {bronce} medalla(s) de bronce,")
            print(f"se espera un total de {int(prediccion[0])} medalla(s).\n")
        
        except ValueError:
            print("\nError: Asegúrese de ingresar tres números separados por comas. Ejemplo: 1,4,3\n")
        
        except KeyboardInterrupt:
            print("\nSaliendo del sistema... ¡Gracias por usar el Sistema de Predicción de Medallas!")
            sys.exit()

# Main
if __name__ == "__main__":
    # Ruta fija del modelo
    ruta_modelo = './modelo_entrenado/olympics_medal_prediction_model.pkl'

    # Cargar el modelo entrenado
    try:
        modelo = cargar_modelo(ruta_modelo)
        print(f"Modelo cargado desde {ruta_modelo}")
    except FileNotFoundError:
        print(f"Archivo de modelo no encontrado en {ruta_modelo}. Asegúrese de que el modelo está entrenado y guardado correctamente.")
        sys.exit(1)

    # Iniciar la interfaz de usuario
    interfaz_usuario(modelo)
