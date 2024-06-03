import numpy as np
import matplotlib.pyplot as plt

# Definimos la función que queremos aproximar
def f(x):
    """
    Función que devuelve un valor basado en el rango en el que se encuentra x.

    Args:
        x (numpy array): Array de valores de entrada.

    Returns:
        numpy array: Array de valores de salida correspondientes a la función.
    """
    x = x.astype(float)
    return np.piecewise(x, [(x >= -10) & (x < -2), (x >= -2) & (x < 0), (x >= 0) & (x <= 10)],
                        [lambda x: -2.186 * x - 12.864,
                         lambda x: 4.246 * x,
                         lambda x: 19 * np.exp(-0.05 * x - 0.5) * np.sin(0.03 * x ** 2 + 0.7 * x)])


# Función para crear y dividir los datos
def create_and_split_data(split_ratio=0.8, n_samples=200, range=(-10, 10)):
    """
    Crea y divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        split_ratio (float, optional): Proporción de los datos que se utilizarán para el entrenamiento. Por defecto es 0.8.
        n_samples (int, optional): Número de muestras a generar. Por defecto es 200.
        range (tuple, optional): Rango de los valores de x. Por defecto es (-10, 10).

    Returns:
        tuple: Tupla que contiene los arrays de x, y, train y test.
    """
    # Creamos los datos de x e y para la función f(x)
    x = np.linspace(range[0], range[1], n_samples)
    y = f(x)
    # Ponemos los datos en un array de 2 columnas para tener la forma (x, y)
    x_y = np.column_stack((x, y))

    # Dividimos los datos en 80% para entrenamiento y 20% para test
    np.random.shuffle(x_y)
    split_index = int(len(x_y) * split_ratio)
    train, test = x_y[:split_index], x_y[split_index:]

    return x, y, train, test

# Función para graficar los datos
def plot_data(x, y):
    """
    Grafica los datos de x e y.

    Args:
        x (numpy array): Array de valores de x.
        y (numpy array): Array de valores de y.
    """
    plt.plot(x, y, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('f(x)')

# Función para establecer la función de activación
def set_activation(activation):
    """
    Establece la función de activación.

    Args:
        activation (str): Nombre de la función de activación.

    Returns:
        function: Función de activación.
    """
    if activation == 'tanh':
        return lambda x: np.tanh(x)
    elif activation == 'relu':
        return lambda x: np.maximum(0, x)
    else:
        return lambda x: 1 / (1 + np.exp(-x))

# Función para establecer la derivada de la función de activación
def set_derivative(activation):
    """
    Establece la derivada de la función de activación.

    Args:
        activation (str): Nombre de la función de activación.

    Returns:
        function: Derivada de la función de activación.
    """
    if activation == 'tanh':
        return lambda x: 1 - np.tanh(x) ** 2
    elif activation == 'relu':
        return lambda x: np.where(x > 0, 1, 0)
    else:
        return lambda x: x * (1 - x)

# Función para dividir un número en cuatro partes
def divide_in_four_parts(n):
    """
    Divide un número en cuatro partes iguales.

    Args:
        n (int): Número a dividir.

    Returns:
        numpy array: Array con los cuatro puntos de división.
    """
    return np.linspace(0, n, 5)

# Clase para la red neuronal multicapa
class MLP:
    """
    Clase para una red neuronal multicapa.

    Attributes:
        n_inputs (int): Número de entradas.
        n_hidden (int): Número de neuronas en la capa oculta.
        n_outputs (int): Número de salidas.
        learning_rate (float): Tasa de aprendizaje.
        epochs (int): Número de épocas.
        activation (function): Función de activación.
        deriv (function): Derivada de la función de activación.
        weights_input_hidden (numpy array): Pesos de la capa de entrada a la capa oculta.
        weights_hidden_output (numpy array): Pesos de la capa oculta a la capa de salida.
        hidden_bias (numpy array): Bias de la capa oculta.
        output_bias (numpy array): Bias de la capa de salida.
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.00001, epochs=1000, activation='sigmoid'):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = set_activation(activation)
        self.deriv = set_derivative(activation)

        self.weights_input_hidden = np.random.randn(n_inputs, n_hidden)
        self.weights_hidden_output = np.random.randn(n_hidden, n_outputs)
        self.hidden_bias = np.random.randn(n_hidden)
        self.output_bias = np.random.randn(n_outputs)

    # Método para la propagación hacia adelante
    def forward(self, x):
        """
        Realiza la propagación hacia adelante.

        Args:
            x (numpy array): Array de entradas.

        Returns:
            tuple: Tupla que contiene el array de salidas y el array de salidas de la capa oculta.
        """
        x = x.reshape(-1, self.n_inputs)
        output_hidden = self.activation(np.dot(x, self.weights_input_hidden) + self.hidden_bias)
        output = np.dot(output_hidden, self.weights_hidden_output) + self.output_bias

        return output, output_hidden

    # Método para la retropropagación
    def backpropagate(self, output, output_hidden, x, target):
        """
        Realiza la retropropagación.

        Args:
            output (numpy array): Array de salidas.
            output_hidden (numpy array): Array de salidas de la capa oculta.
            x (numpy array): Array de entradas.
            target (numpy array): Array de salidas deseadas.
        """
        target = target.reshape(-1, self.n_outputs)
        error_output = target - output
        delta_output = error_output

        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.deriv(output_hidden)

        self.weights_hidden_output += np.dot(output_hidden.T, delta_output) * self.learning_rate
        self.output_bias += np.sum(delta_output, axis=0) * self.learning_rate
        self.weights_input_hidden += np.dot(delta_hidden.T, x) * self.learning_rate
        self.hidden_bias += np.sum(delta_hidden, axis=0) * self.learning_rate

    # Método para entrenar la red
    def train(self, x, y, x_train, y_train, x_test, y_test, doPlot=False):
        """
        Entrena la red neuronal.

        Args:
            x (numpy array): Array de entradas.
            y (numpy array): Array de salidas deseadas.
            x_train (numpy array): Array de entradas de entrenamiento.
            y_train (numpy array): Array de salidas deseadas de entrenamiento.
            x_test (numpy array): Array de entradas de prueba.
            y_test (numpy array): Array de salidas deseadas de prueba.
            doPlot (bool, optional): Si es True, se grafican los resultados. Por defecto es False.
        """
        epochs_to_plot = np.linspace(0, self.epochs, 6)
        epochs_to_plot = epochs_to_plot.astype(int)
        plot_index = 1

        for actualEpoch in range(self.epochs + 1):
            output, output_hidden = self.forward(x_train)
            self.backpropagate(output, output_hidden, x_train, y_train)

            if doPlot and actualEpoch in epochs_to_plot:
                test_output = self.forward(x_test)[0]

                plt.subplot(2, 3, plot_index)
                plt.plot(x, y, 'b')
                plt.scatter(x_test, y_test, c='g')
                plt.scatter(x_test, test_output, c='r')
                plt.title(f"Epoch {actualEpoch}")
                plot_index += 1

        if doPlot:
            plt.tight_layout()
            plt.show()


# Creo y divido los datos
x, y, train, test = create_and_split_data()

# Divido en x e y los datos de entrenamiento y prueba
train_x = train[:, 0]
train_y = train[:, 1]
test_x = test[:, 0]
test_y = test[:, 1]

# Configuraciones que voy a evaluar
learning_rates = [0.01, 0.001, 0.0001]
hidden_sizes = [2, 10, 20, 50]
activations = ['sigmoid', 'relu']

best_error = np.inf
best_params = None

# Compruebo cual es la mejor configuración de la red neuronal
for learning_rate in learning_rates:
    for hidden_size in hidden_sizes:
        for activation in activations:
            print(f"Training with Learning Rate={learning_rate}, Hidden Size={hidden_size}, Activation={activation}...")
            nn = MLP(1, hidden_size, 1, learning_rate=learning_rate, epochs=20000, activation=activation)
            nn.train(x, y, train_x, train_y, test_x, test_y)
            output = nn.forward(test_x)[0]
            output = np.squeeze(output)
            error = np.mean((output - test_y) ** 2)
            if error < best_error:
                best_error = error
                best_params = (learning_rate, hidden_size, activation)

# Imprimo la mejor configuración
print(f"\nBest parameters: Learning Rate={best_params[0]}, Hidden Size={best_params[1]}, Activation={best_params[2]}, "
      f"Error={best_error}")

# Utilizo la mejor configuración encontrada previamente
# Haciendo varias pruebas he visto que alrededor de las 40000 épocas se obtiene una aproximación bastante buena
# Sobre todoo para aproximar la última parte de la función
best_nn = MLP(1, best_params[1], 1, learning_rate=best_params[0], epochs=40000, activation=best_params[2])
best_nn.train(x, y, train_x, train_y, test_x, test_y, doPlot=True)
