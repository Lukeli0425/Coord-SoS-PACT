function SOS = waterSOSfromTemperature(T)

    a = ...
    [...
      1.402385e3; 
      5.038813;
     -5.799136e-2;
      3.287156e-4;
     -1.398845e-6;
      2.787860e-9;];

    SOS = (T.^[0, 1, 2, 3, 4, 5]) * a;
end