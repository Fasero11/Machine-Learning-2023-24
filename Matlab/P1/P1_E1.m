% Ejercicio 1. Regresión Lineal
% Julia López
% Gonzalo Vega
% AA - 2023

clc
clear

x1 = [1,2,3,4,5,6,7,8,9,10];
x2 = [1,1,1,1,1,1,1,1,1,1];
X = [x1;x2].';

Y = [0, 0.69, 1.1, 1.39, 1.61, 1.79, 1.95, 2.08, 2.2, 2.3];

% --- 1.1 ---
% Crear modelo de regresión lineal
mdl = fitlm(X,Y)

%% 

% --- 1.2 ---
% Calcular predicciones
ye = predict(mdl, X);

% Mostrar datos iniciales
scatter3(x1, x2 ,Y, 'b*');
hold on 
% Mostrar predicción
plot3(x1,x2, ye, 'r');
hold off

%% 

% --- 1.4 ---
% Obtener función de coste (J)
w0 = 0.24
w1 = 0.23
w2 = 0
J = 0
