function J = ComputeCostMulti(Thj,X,y)
% Esta es la funcion de coste, le entran las thetas actuales y los datos X

    m=size(X,1);       
    h = (Thj*X')';% esto es igual a h=theta0+theta1*x, la recta pero operada matricialmente
    
    J=(1/(2*m))*sum((h - y).^2);%funcion de coste nos devuelve el error medio de la aproximacion
   
end

