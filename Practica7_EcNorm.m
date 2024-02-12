clear all
close all
clc 

%{
CONTENIDOS: Respresentacion 3D de los datod, plot3

El objetivo de esta practica es implemetar un algoritmo de regresion
lineal multivariable para predecir el precio de viviendas. 

Para ello nos dan un fichero ex1data2.txt con todos los datos, en donde la
primera columna es el tamaño de la casa en pies cuadrados, la segunda
columna es el numero de habitaciones y la tercera es el precio de la casa,
la segunda columna es el nuemro de habitaciones d ela casa y la tercera es
el precio de la casa. 

%}

%% Cargamos los datos

data=load('ex1data2.txt'); %cargamos los datos del fichero con load

% extraemos los datos en 2 vectores x1,x2,z para hacerlo generico

x1=data(:,1);%tamaño de la casa (Datos de entrenamiento)
x2=data(:,2);%numero de habitaciones (Datos de entrenamiento)
y=data(:,3);%beneficio de los restaurantes (Etiquetas o Datos de salida) SERIA NUESTRA Z

%% Pintar los datos en una grafica 3D 

%plot de los datos con plot3

figure()
plot3(x1, x2, y,'x'); %Ojo usar traspuesta
xlabel('Tamanno (pies cuadrados)');
ylabel('Numero de habitaciones');
zlabel('Precio (€)');
grid on
title('Datos ex1data2.txt');


%% Ecuacion normal

%{
Existe una forma de calcular los thetas sin necesidad de usar descenso por
gradiente, y es con una operacion matricial. Tiene menos pasos y es más
simple pero no es escalable, si tenemos muchos datos de entrada el coste
computacional será muy alto.
%}

%Caso con los datos originales sin normalizar
X=horzcat(ones(size(x1,1),1),x1,x2);% solo añado x1 porque mis datos de entrenamiento son x, los datos de entrada
[Theta_vector]  = NormalEqn(X, y);

%Caso con los datos normalizados
X_norm = FeatureNormalize(X);
[Theta_vector_norm]  = NormalEqn(X_norm, y);% calcula los thetas a traves de la ecuacion normal

% Ec de h con estos Thetas TUNEAR PARA QUE VALGA PARA CUALQUIER NUMERO DE
% THETA

h = Theta_vector_norm(1) + Theta_vector_norm(2)*x1 + Theta_vector_norm(3)*x2;

%Representacion 3D REVISAAAR!!!!

% NOTA: si no te piden representar los normalizados cambiar X_norm por X

x1_aux = linspace(min(x1), max(x1), 100);
x2_aux = linspace(min(x2), max(x2), 100);

T=Theta_vector;

for i = 1:100
    for j = 1:100
        Z(i, j) = T(1) + T(2)*x1_aux(i) + T(3)*x2_aux(j);
    end
end

% figure()
% plot3(X(:,2),X(:,3),y,'rx')
% hold on
% [XX,YY]=meshgrid(min(X(:,2)):100:max(X(:,2)), min(X(:,3)):1:max(X(:,3)));
% Z = Theta_vector(1)+ Theta_vector(2).*XX +Theta_vector(3).*YY;
% surf(XX,YY,Z)
% grid on

figure()
plot3(X(:,2),X(:,3),y,'rx')
hold on
surf(x1_aux,x2_aux,Z')
grid on
xlabel('Tamanno(pies cuadrados)');
ylabel('Habitaciones');
zlabel('Precios(€)');
title('Plano que mejor ajusta los datos + datos(cruces)');

%{
PREGUNTAS

¿Que valor del vector theta se obtiene en ambos casos (datos originales
y normalizados)?

Como podemos ver en las variables Theta_vector y Theta_vector_norm, el
valor de las thetas obtenidas por el algoritmo es diferente

Theta_vector=[89597.9095427972,139.210674017626,-8738.01911232770]
Theta_vector_norm=[340412.659574468,110631.050278846,-6649.47427081976]

Podemos ver como los valores del vector de thetas normalizado es mayor que
el de Theta_vector sin normalizar


¿Coincide con el valor obteniendo utilizando descenso por gradiente?

Únicamente en el caso de ThetaVector

¿Por qué cree que es esto?

Imagino que debido a la normalizacion de las variables de entrada

%}



