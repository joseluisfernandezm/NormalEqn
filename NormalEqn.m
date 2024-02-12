function [Theta_vector]  = NormalEqn(X, y)

    % La funcion recibe una matriz de datos y el vector de etiquetas
    % ya la salida son los thetas

    % Theta_vector vector fila
    
    Theta_vector = ((X'*X)^(-1))*X'*y;%formula ecuacion normal
    Theta_vector = Theta_vector';%devuelvo traspuesto
    
end