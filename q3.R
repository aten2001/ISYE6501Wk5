library('FrF2')

feature_names <- c('Feature 1','Feature 2','Feature 3','Feature 4','Feature 5', 
                   'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10')

design <- FrF2(nruns = 16, factor.names = feature_names)

design_frame <- data.frame(design)

for (varname in names(design_frame)) {
    design_frame[,varname] <- factor(design_frame[,varname], levels = c(-1, 1), 
                                   labels = c('No', 'Yes'))
}

