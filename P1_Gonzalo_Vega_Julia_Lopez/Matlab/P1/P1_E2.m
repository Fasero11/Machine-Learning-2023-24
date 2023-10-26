% Ejercicio 2. Regresión Logística Binaria
% Julia López
% Gonzalo Vega
% AA - 2023

clc
clear

x1 = [0.89,0.41,0.04,0.75,0.15,0.14,0.61,0.25, ...
    0.32,0.40,1.26,1.68,1.23,1.46,1.38,1.54,1.99,1.76,1.98,1.23];
x2 = [0.41,0.39,0.61,0.17,0.19,0.09,0.32,0.77, ...
    0.23,0.74,1.53,1.05,1.76,1.60,1.86,1.99,1.93,1.41,1.00,1.54];
x3 = [0.69,0.82,0.83,0.29,0.31,0.52,0.33,0.83, ...
    0.81,0.56,1.21,1.22,1.33,1.10,1.75,1.75,1.54,1.34,1.83,1.55];

X = [x1;x2;x3].';
% + = 1; o = 0
Y = [1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,0];

% --- 2.1 ---
% Crear modelo de regresión logística binaria
mdl = fitglm(X,Y,'Distribution','binomial')

%% 

% --- 2.2 ---
% representa las entradas
% Crear una figura para visualizar los datos y las predicciones

% Visualización de datos y predicciones
figure;
predictions = predict(mdl, X);

for id = 1:length(predictions)
    if Y(1,id) == 0 && predictions(id,1) < 0.5
        % círculo acierto
        scatter3(X(id, 1), X(id, 2), X(id, 3), 'o', 'filled', 'MarkerFaceColor', 'b');
    elseif Y(1,id) == 0 && predictions(id,1) > 0.5
        % círculo fallo
        scatter3(X(id, 1), X(id, 2), X(id, 3), 'o', 'filled', 'MarkerFaceColor', 'r');
    elseif Y(1,id) == 1 && predictions(id,1) > 0.5
        % cruz acierto
        scatter3(X(id, 1), X(id, 2), X(id, 3), '+', 'MarkerEdgeColor', 'b');
    elseif Y(1,id) == 1 && predictions(id,1) < 0.5
        % cruz fallo
        scatter3(X(id, 1), X(id, 2), X(id, 3), '+', 'MarkerEdgeColor', 'r');
    end
    hold on;
end


% Ajustes de la figura
title('Clasificador de Regresión Logística');
xlabel('Característica 1');
ylabel('Característica 2');
zlabel('Característica 3');

grid on;


%% 

% Calcular el error en porcentaje sobre los datos de entrenamiento
train_predictions = predict(mdl, X);


% Convertir las predicciones en etiquetas binarias (0 o 1)
predicted_classes = round(train_predictions);

% Calcular el número de predicciones que coinciden
eq = 0;
for i = 1:length(predicted_classes)
   if Y(i) == predicted_classes(i)
       eq = eq + 1;
   end
end


%Calcular el error obtenido
err = ((length(predicted_classes) - eq) / length(predicted_classes)) *100;
 
disp("El porcentaje de error de los datos de entrenamiento es: " +err+ "%")


