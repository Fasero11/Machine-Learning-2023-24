clc
clear

% Define las variables
conceder = ["SI","SI","NO","SI","NO","SI","NO","SI","SI","SI"];
trabajo = [0,1,0,0,1,0,1,1,0,1];
ingresos = [2,1,1,0,2,0,0,1,1,1];
asnef = [0,0,1,0,1,0,0,0,0,0];
cantidad = [2,1,1,0,2,1,0,1,2,0];

% Muestra los resultados
fprintf('Entropía de Trabajo: %f\n', entropy_by_feature(trabajo, conceder));
fprintf('Entropía de Ingresos: %f\n', entropy_by_feature(ingresos, conceder));
fprintf('Entropía de Asnef: %f\n', entropy_by_feature(asnef, conceder));
fprintf('Entropía de Cantidad: %f\n', entropy_by_feature(cantidad, conceder));

% Función para calcular la entropía condicional
function entropy = entropy_by_feature(feature, target)
    unique_values = unique(feature);
    entropy = 0;
    
    for i = 1:length(unique_values)
        value = unique_values(i);
        [m, num_muestras] = size(feature);
        num_SI = sum(target(feature == value) == "SI");

        prob_SI = num_SI / num_muestras;
        if prob_SI > 0 && prob_SI < 1
            entropy = entropy - prob_SI * log2(prob_SI) - (1 - prob_SI) * log2(1 - prob_SI);
        end
    end
end