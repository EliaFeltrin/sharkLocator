function [datasetTable] = tfDatasetParser(datasetPath,nImageToLoad,firstClmLabel, secondClmLabel)
    xmlpath = strcat(datasetPath, "/annotations/xmls/");
    xmls = dir(xmlpath);

    sz = [nImageToLoad, 2];
    varTypes = ["string", "cell"];
    varNames = [firstClmLabel, secondClmLabel];
    datasetTable = table('Size', sz, 'VariableTypes', varTypes, 'VariableNames',varNames);
    
    for i = 3:nImageToLoad+3 

        filename = "shark_" + int2str(i-2) + ".xml";
        struct = readstruct(strcat(xmlpath, filename)); 
      
        bb(1) = struct.object.bndbox.xmin;
        bb(2) = struct.object.bndbox.ymin;
        bb(3) = struct.object.bndbox.xmax - struct.object.bndbox.xmin;
        bb(4) = struct.object.bndbox.ymax - struct.object.bndbox.ymin;

        datasetTable(i-2,:) = {{convertStringsToChars(strcat(pwd, "/", datasetPath, "/images/", struct.filename))}, {bb}};

    end
end