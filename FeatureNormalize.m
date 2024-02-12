function Xnorm = FeatureNormalize(X)
    % X matriz de num_datos x Num_variables
    
    X_aux = X(:, 2:end);%extraigo las columnas con los datos de entrada
    num_variables = size(X_aux, 2);
    % Inicializar
    Media = zeros(1, num_variables);
    Sigma = zeros(1, num_variables);
    X_norm_aux = X_aux.*0;
    
    for i = 1:num_variables
        Media(i) = mean(X_aux(:, i));

        Sigma(i) = std(X_aux(:, i));
        
        X_norm_aux(:, i) = (X_aux(:, i) - Media(i))/Sigma(i);
    end
    Xnorm = [X(:, 1), X_norm_aux];%vuelvo a incluir la columna de 1
end