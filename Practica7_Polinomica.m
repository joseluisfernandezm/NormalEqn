clear all
close all
clc 

%{
CONTENIDOS: Respresentacion 3D de los datod, plot3

En teoria hemos visto que se pueden aplicar todas las herramientas
estudiadas para la regresion lineal a problemas en los que se quiera
ajustar una ecuacion de hipotesis hTheta, que no se corresponda con una
ecuacion lineal sino con la de un polinomio.

%}

%% Cargamos los datos

data=load('ex1data1.txt'); %cargamos los datos del fichero con load

% extraemos los datos en 2 vectores x1,x2,z para hacerlo generico

x1=data(:,1);%tamaño de la casa (Datos de entrenamiento)
x2=x1.^2;
% x3=x1.^3;
y=data(:,2);%beneficio de los restaurantes (Etiquetas o Datos de salida) SERIA NUESTRA Z


%% Regresion polinomica

X=horzcat(ones(size(x1,1),1),x1,x2);%añadir aqui mas xi si es necesario
[Theta_vector]=NormalEqn(X, y);

%Representacion 2D ya que es univariable, solo tenemos una variable de
%entrada

x1_aux = linspace(min(x1), max(x1), 100);
x2_aux = x1_aux.^2;

h = Theta_vector(1) + Theta_vector(2)*x1_aux + Theta_vector(3)*x2_aux;


J_pol = ComputeCost(Theta_vector, X, y);
disp(J_pol);

figure()
plot(x1, y, 'x');
xlabel('Habitantes (en 10k)');
ylabel('Beneficio (en 10k de euros)');
hold on
plot(x1_aux, h);
title('Funcion h, aproximacion a y');
hold off


%{

PREGUNTAS

¿Que valor del vector theta ha obtenido?

¿Cual es el valor de la funcion de coste para ese vector?

¿Es el valor de la funcion de coste mayor o menor que el obtenido en la
 practica 6 para el caso de los datos con una recta?

%}


