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

%% Descenso por gradiente
%{
Implementamos el algoritmo de desdenso por gradiente

Primero almacenamos los datos de entrenamiento en una matiz X, cada fila
va a ser un dato y añadiremos una columna de todo unos que representa Xo

%}

% X=[ones(size(data,1),1),x1,x2];%OJO CON TRASPONER, se puede hacer asi o con la funcio horzcat
% % X=X';

X=horzcat(ones(size(x1,1),1),x1,x2);% solo añado x1 porque mis datos de entrenamiento son x, los datos de entrada

Thj=[0,0,0]; %inicializamos los theta a 0

alpha=3*10^-7;  % fijamos un coeficiente de aprendizaje
% alpha=1;%
n_iters=200;% numero de iteraciones, nos lo dan

[J,Thetas_finales,Tmat] =  GradientDescentMulti(X,y,alpha,Thj,n_iters);

figure()
plot(J);
title(['alpha =', num2str(alpha), ' Evolucion de J(Theta)'])
xlabel('N iters')
ylabel('Coste J(Theta)')

J_final=ComputeCostMulti(Thetas_finales,X,y);%para contestar pregunta 2 de abajo


%{

- Preguntas: 

¿Cual es el valor final de los parametros thetaj para el mejor valor de 
aloha encontrado?

para un alpha de 3*10^-7 el vector de thetas finales es

[0.64,1.65e+02,1.23]

¿Cual es el valor final de la funcion de coste para esre caso?

Eso viene en Jfinal, calculada con las thetas calculadas al final del
algoritmo de descenso del gradiente, y es 2.4

¿El algoritmo converge para cualquier valor de alpha?

Para un alpha igual a 1 el algoritmo diverge, así que este no converge para
cualquier valor de alpha. Esto lo hemos comprobado poniendo el alpha a 1 y
viendo como la funcion de coste diverge en la grafica. Otra forma de
comprobarlo sería debuggeando la funcion de descenso de gradiente y viendo
como la funcion de coste aumenta sin parar.

PREGUNTAR A CHAT SI TAMBIEN SE DEBE POR LAS THETAS O SOLO POR LA FUNCION DE
COSTE QUE DIVERJA LA FUNCION.



%}



%% Normalizacion de variables (features)

%{
Al tener variables cuyos valores estan en rangos muy diferentes, la
normalizacion puede ayudar a que el algoritmo de descenso por graciente
converja más rápidamente.
%}

X=horzcat(ones(size(x1,1),1),x1,x2);% solo añado x1 porque mis datos de entrenamiento son x, los datos de entrada
X_norm = FeatureNormalize(X);%funcion que nos normaliza los datos de entrada dejandolos entre 0 y 1

Thj=[0,0,0]; %inicializamos los theta a 0

alpha=1;  % fijamos un coeficiente de aprendizaje

n_iters=200;% numero de iteraciones, nos lo dan

[J_norm,Thetas_finales_norm,Tmat_norm] =  GradientDescentMulti(X_norm,y,alpha,Thj,n_iters);

figure()
plot(J_norm);
title(['alpha =', num2str(alpha), ' Xnorm: Evolucion de J(Theta)'])
xlabel('N iters')
ylabel('Coste J(Theta) Normalizando')

J_final=ComputeCostMulti(Thetas_finales,X,y);%para contestar pregunta 2 de abajo


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

%% Regresion polinomica



