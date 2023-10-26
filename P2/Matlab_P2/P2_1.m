% P2 - Ejercicio 1. Árboles de decisión
% Julia López
% Gonzalo Vega
% AA - 2023

clc
clear

conceder = ["SI","SI","NO","SI","NO","SI","NO","SI","SI","SI"];
Y = conceder.';

%.%.%.%. ONE HOT ENCODING %.%.%.%.
%{
trabajo = ["fijo","temporal","fijo","fijo","temporal","fijo","temporal","temporal","fijo","temporal"];
ingresos = ["altos","medios","medios","bajos","altos","bajos","bajos","medios","medios","medios"];
asnef = ["no","no","si","no","si","no","no","no","no","no"];
cantidad = ["alta","media","media","baja","alta","media","baja","media","alta","baja"];


trabajo = categorical(trabajo);
ingresos = categorical(ingresos);
asnef = categorical(asnef);
cantidad = categorical(cantidad);

trabajo = onehotencode(trabajo,1).';
ingresos = onehotencode(ingresos,1).';
asnef = onehotencode(asnef,1).';
cantidad = onehotencode(cantidad,1).';

X = [trabajo, ingresos, asnef, cantidad];
%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.
%}

trabajo = [0,1,0,0,1,0,1,1,0,1].';
ingresos = [2,1,1,0,2,0,0,1,1,1].';
asnef = [0,0,1,0,1,0,0,0,0,0].';
cantidad = [2,1,1,0,2,1,0,1,2,0].';

X = [trabajo, ingresos, asnef, cantidad];

% --- 1.1 ---
% Crear árbol de decisión

mdl = fitctree(X, Y, 'MinParentSize', 1);

view(mdl,'Mode','Graph')


