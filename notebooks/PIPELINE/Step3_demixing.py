#Initialize packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import nnls
from scipy import ndimage
import spectrochempy as scp
from skimage import restoration
from PIL import Image
from pathlib import Path
import os
import pickle

#Define function for making guess spectra in MCR using SIMPLISMA
def guessSpectra(X_flatten,nchan):
    #Input linearized image data, with image pixels in x direction and wavelengths in y direction
    simpl = scp.SIMPLISMA(n_components=nchan, tol=0.2, noise=3, log_level="INFO")
    simpl.fit(X_flatten.T)
    _ = simpl.C.T.plot(title="Concentration")
    __ = simpl.Pt.plot(title="Pure profiles")
    ___ = simpl.St.plot(title="Spectra")
    #_ = simpl.components.plot(title="Pure profiles")
    #simpl.plotmerit(offset=0, nb_traces=nchan, title="SIMPLISMA merit function")
    return simpl.components, simpl.C, simpl.Pt, simpl.St

#Define function for MCR-ALS

#Required input: background subtracted stack of image data in .npy format in order of depth, if looking at multiple depths
#outputs: mcr object, flattened raw data, concentration information in linear form, concentration map, spectral fits (constrained and not constrained)
#if waterSpec is True, will also return the water spectrum as the final spectrum in St and St_unconstrained
def fit_mcrals(inputData, St_guess, ndepth, nfilt, nchan, specMethod, waterSpec):
    #rawData = np.delete(rawData,4,1)   #Delete undesired filters
    Xshape = inputData.shape
    X_flatten = inputData.reshape(-1, inputData.shape[-1]) #reduce image to 1 dimension
    
    #choose no normalization, euclid norm (spctral integral = 1), or spectral max norm; comment out the other two options
    #Other options located at https://www.spectrochempy.fr/latest/reference/generated/spectrochempy.MCRALS.html#spectrochempy.MCRALS
    #mcr = scp.MCRALS(log_level="INFO")
    #mcr = scp.MCRALS(log_level="INFO",normSpec="euclid")        solverSpec = "nnls", solverConc = "nnls",
    
    #Using non negative least squares regression on spectra and concentrations since these profiles since they are always positive. 
    #Will need to use pnnls when accounting for water spectra
    #Autofluorescence would constitute a third component
    #Using non-negativity and unimodality constraints too because of fluorescence behaviors
    
    mcr = scp.MCRALS(log_level="INFO", 
                     nonnegConc="all",nonnegSpec="all", normSpec = None,
                     unimodSpec="all", unimodSpecMod = "strict",unimodSpecTol=1.1,
                     solverSpec = "nnls", solverConc = "nnls",   tol = 0.1,
                     maxdiv = 10, max_iter = 20)
    print(X_flatten.shape)

    mcr_out = np.empty((ndepth,1), dtype=object)
    mcr_conc = np.empty([Xshape[0]*Xshape[1],nchan,ndepth])
    St_fit = np.empty([nchan,nfilt,ndepth])
    St_fit_nonConstrained = np.empty([nchan,nfilt,ndepth])
    
    #Fit MCR-ALS
    if specMethod == "SIMPLISMA":
        SIMPL_Pt_fit = np.empty([nchan,nfilt,ndepth])
        SIMPL_St_fit = np.empty([nchan,nfilt,ndepth])
        if waterSpec == True:
            nchan = nchan+1
        
        #MCRALS with SIMPLISMA
        for i in range(0,ndepth):
   
            SIMPLISMAcomp, SIMPLISMAconc, SIMPLpt,SIMPLst = guessSpectra(X_flatten[...,(nfilt*i):(nfilt+nfilt*i)].T,nchan)
            #print(SIMPLISMAconc)
            #SIMPLdataframe = pd.DataFrame(data = SIMPLISMAconc)
            #SIMPLdataframe.columns = St_guess.columns
            #SIMPLdataframe.index = St_guess.index

            #for j in range(0,nchan):
            #    SIMPLdataframe.loc[:,St_guess.columns[j]] = SIMPLISMAfit.C[:,j]
            print(SIMPLISMAconc.dtype) 
            print(SIMPLISMAconc.values)
            mcr_out[i] = mcr.fit(X_flatten[...,(nfilt*i):(nfilt+nfilt*i)],SIMPLISMAcomp)    
    
            mcr_conc[...,i] = np.array(mcr_out[i,0].C)
            print(mcr_conc.shape) 
 
            #Store and plot model spectra
            #mcr_out[i,0].St_unconstrained.plot()
            #mcr_out[i,0].St.plot()

            SIMPL_Pt_fit[...,i] = SIMPLpt
            SIMPL_St_fit[...,i] = SIMPLst
            St_fit[...,i] = mcr_out[i,0].St
            St_fit_nonConstrained[...,i] = mcr_out[i,0].St_unconstrained

    elif specMethod == "PURE SPECTRA":
        #If water spectra desired, add it as a 3rd component to spectra guess. load the water spectrum
        waterSpectra = np.zeros([nfilt,1])
        if waterSpec == True:
            St_guess = pd.DataFrame(data = np.concatenate((St_guess,waterSpectra),axis = 1))
        #MCR-ALS with pure spectra
        for i in range(0,ndepth):
            if isinstance(St_guess, np.ndarray):
                mcr_out[i] = mcr.fit(X_flatten[...,(nfilt*i):(nfilt+nfilt*i)],St_guess)
            else:
                print(X_flatten[...,(nfilt*i):(nfilt+nfilt*i)].shape)
                mcr_out[i] = mcr.fit(X_flatten[...,(nfilt*i):(nfilt+nfilt*i)],St_guess.to_numpy())    
    
            mcr_conc[...,i] = np.array(mcr_out[i,0].C)
            print(mcr_conc.shape) 
 
            #Store and plot model spectra
            mcr_out[i,0].St_unconstrained.plot()
            mcr_out[i,0].St.plot()

            St_fit[...,i] = mcr_out[i,0].St
            St_fit_nonConstrained[...,i] = mcr_out[i,0].St_unconstrained
    
    mcr_image = np.squeeze(np.reshape(mcr_conc, [Xshape[0],Xshape[1],nchan,ndepth]))
    return [mcr_out, X_flatten, mcr_conc, mcr_image,St_fit,St_fit_nonConstrained]

#Define function for plotting images
def make_image(ax_in, concentrations, reshape):
    ps = np.reshape(concentrations, reshape)
    ax_in.imshow(ps, cmap='gray')

def binarize_and_measure(img, threshold, min_size, depths):

    # Initialize an empty list to store detected objects
    #detected_objects = np.empty(objectStoreSize, dtype=DetectedObject) #Detect a maximum of 200 objects
    filled_masks = np.zeros([img.shape[0],img.shape[1],len(depths)])
    labeled_objects = np.zeros([img.shape[0],img.shape[1],len(depths)])
    #initialize centroid, size, depth, depthno, objectno
    centroid = np.full(shape=[200,2], fill_value=np.nan)
    size = np.full(shape=200, fill_value=np.nan)
    depthinfo = np.full(shape=200, fill_value=np.nan)
    depthno = np.full(shape=200, fill_value=np.nan)
    coords = np.empty([200], dtype=object)
    objectno = np.full(shape=200, fill_value=np.nan)
    thresholds = np.full(shape=[200,img.shape[2]], fill_value=np.nan)

    c = 0
    for i in range(0,(len(depths))):
        imgset = img[...,i]
        print(imgset.shape)
        #find agreement between contrast agent image channels based on their union
        
        # Binarize the image based on the threshold in each channel. Merge into a single binary mask based on the union of target channels
        print(np.amax(imgset))
        binary_mask = np.zeros(imgset.shape)
        
        #Below: binary mask based on threshold
        #for j in range(0,imgset.shape[2]):
        #    binary_mask[...,j] = imgset[...,j] > threshold*np.amax(imgset[...,j])

        #Below: binary mask based on mat2gray and binarization
        thres_store = np.empty(imgset.shape[2])
        for j in range(0,imgset.shape[2]):
            #grayimg = img_as_ubyte(imgset[...,j])
            #print("Max and min of mask are",[np.amax(grayimg),np.amin(grayimg)])
            #print(grayimg.shape)
            data= np.array(imgset[...,j],dtype=float)
            rescaled_data = data/np.amax(data)
            thres = threshold_otsu(rescaled_data)
            print("Threshold is",thres)
            binary_mask[...,j] = rescaled_data > thres
            thres_store[j] = thres  

        combine_mask = np.sum(binary_mask,axis=2) > 0
        # Fill holes in the binary mask
        filled_mask = ndimage.binary_fill_holes(combine_mask)
        #Close gaps in the mask and connect close regions. Test using ndimage.binary_closing and ndimage.binary_propagation
        closed_mask = ndimage.binary_closing(filled_mask, structure=np.ones((5,5)))
        #store mask
        filled_masks[:,:,i] = closed_mask
        # Label connected components in the filled mask
        labeled_object, num_objects = ndimage.label(closed_mask)
        # Measure properties of labeled objects using skimage.measure.regionprops
        objects_props = measure.regionprops(labeled_object)
        labeled_objects[:,:,i] = labeled_object
        # Remove small objects based on the min_size threshold. store size and centroid information
        for obj_prop in objects_props:
            sizecheck = obj_prop.area 
            np.empty((ndepth,1), dtype=object)
            if sizecheck >= min_size:
                centroid[c,0]= round(obj_prop.centroid[0])
                centroid[c,1]= round(obj_prop.centroid[1])

                thresholds[c] = thres_store
                size[c] = sizecheck
                depthinfo[c] = depths[i]
                depthno[c] = i
                coords[c] = np.empty(2, dtype=object)
                coords[c] = obj_prop.coords
                objectno[c] = c
                c = c+1   

    # Create an object to store centroid and size
    #prune objects with no values
    # Create a boolean mask based on valid centroid entries
    valid_mask = ~np.isnan(objectno)
    print(valid_mask)

    # Filter out the invalid entries across all attributes
    centroid = centroid[valid_mask]
    size = size[valid_mask]
    depthinfo = depthinfo[valid_mask]
    depthno = depthno[valid_mask]
    objectno = objectno[valid_mask]
    coords = np.array(coords)[valid_mask].tolist()

    # Initialize pandas dataframe to store object information
    objectData = pd.DataFrame(data={'centroidy': centroid[:,0], 
                                    'centroidx': centroid[:,1], 
                                    'size': size, 
                                    'depth': depthinfo, 
                                    'depthno': depthno,
                                    'objectcoord': coords},
                                index = objectno)

    print(objectData.loc[:,"centroidy"])
    print(objectData.loc[:,"centroidx"])
    print(objectData.loc[:,"size"])
    print(objectData.loc[:,"depth"])
    print("Total objects found = ",c)
    
    backgroundMask = filled_masks == 0
    return filled_masks, backgroundMask, labeled_objects, objectData


def grabSignals(modelSignal, modelBackground, objectData, targ_chan, ref_chan, BPref, CNRPref, depthcorr):  
    #Initialize data structures
    targMean = np.empty(len(objectData.loc[:,"centroidx"]))
    targStdev = np.empty(len(objectData.loc[:,"centroidx"]))
    refMean = np.empty(len(objectData.loc[:,"centroidx"]))
    refStdev = np.empty(len(objectData.loc[:,"centroidx"]))
    backTargMean = np.empty(len(objectData.loc[:,"centroidx"]))
    backTargStdev = np.empty(len(objectData.loc[:,"centroidx"]))
    backRefStdev = np.empty(len(objectData.loc[:,"centroidx"]))
    backRefMean = np.empty(len(objectData.loc[:,"centroidx"]))
    targCNR = np.empty(len(objectData.loc[:,"centroidx"]), dtype=object)
    refCNR = np.empty(len(objectData.loc[:,"centroidx"]), dtype=object)
    BP = np.empty(len(objectData.loc[:,"centroidx"]))
    CNRP = np.empty(len(objectData.loc[:,"centroidx"]))
    BPdev = np.empty(len(objectData.loc[:,"centroidx"]))
    CNRPdev = np.empty(len(objectData.loc[:,"centroidx"]))
    targCNRmean = np.empty(len(objectData.loc[:,"centroidx"]))
    refCNRmean = np.empty(len(objectData.loc[:,"centroidx"]))
    targCNRstd = np.empty(len(objectData.loc[:,"centroidx"]))
    refCNRstd = np.empty(len(objectData.loc[:,"centroidx"]))
    BPRMSE = np.empty(len(objectData.loc[:,"centroidx"]))
    CNRPRMSE = np.empty(len(objectData.loc[:,"centroidx"]))

    depthno_pull = objectData.loc[:,"depthno"]
    depth_pull = objectData.loc[:,"depth"]
    coord_pull = objectData.loc[:,"objectcoord"]
    ndepth = pd.unique(depthno_pull).shape[0]

    objectCount = len(objectData.loc[:,"centroidx"])/ndepth

    n = 0
    #Loop through objects
    for i in range(0,len(objectData.loc[:,"centroidx"])):
        coord_object = coord_pull[i].astype(int)
        pixvaltarg = np.empty(len(coord_object))
        pixvalref = np.empty(len(coord_object))
        pixCNRtarg = np.empty(len(coord_object))
        pixCNRref = np.empty(len(coord_object))
        pixBP = np.empty(len(coord_object))
        pixCNRP = np.empty(len(coord_object))
        pixBPerror = np.empty(len(coord_object))
        pixCNRPerror = np.empty(len(coord_object))

        #Extract pixel values
        for j in range(0,len(coord_object)):
            pixvaltarg[j] = modelSignal[coord_object[j,0],coord_object[j,1],targ_chan,int(depthno_pull[i])]
            pixvalref[j] = modelSignal[coord_object[j,0],coord_object[j,1],ref_chan,int(depthno_pull[i])]
        
        #Gather basic population information
        targMean[i] = np.nanmean(pixvaltarg)
        targStdev[i] = np.nanstd(pixvaltarg)
        refMean[i] = np.nanmean(pixvalref)
        refStdev[i] = np.nanstd(pixvalref)
        backTargMean[i] = np.nanmean(modelBackground[:,:,ref_chan, int(depthno_pull[i])])
        backTargStdev[i] = np.nanstd(modelBackground[:,:,ref_chan, int(depthno_pull[i])])
        backRefStdev[i] = np.nanstd(modelBackground[:,:,ref_chan, int(depthno_pull[i])])
        backRefMean[i] = np.nanmean(modelBackground[:,:,ref_chan, int(depthno_pull[i])])

        #calculate per pixel CNR, BP, CNRP in each object
        for j in range(0,len(coord_object)):
            pixCNRtarg[j] = (pixvaltarg[j]-backTargMean[i])/backTargStdev[i]
            pixCNRref[j] = (pixvalref[j]-backRefMean[i])/backRefStdev[i]
            
            if np.size(np.array(depthcorr))>1:#If depth correction matrix is present, apply it
                pixCNRtarg[j] = pixCNRtarg[j]/depthcorr[int(objectData.loc[i,"depthno"])]

            if pixCNRtarg[j] < 1:
                pixCNRtarg[j] = float("NaN")
            if pixCNRref[j] < 1:
                pixCNRref[j] = float("NaN")
            
            pixBP[j] = np.divide((pixCNRtarg[j] - pixCNRref[j]),pixCNRref[j])
            if np.size(np.array(BPref))>1:#If BP reference matrix is present, determine error
                pixBPerror[j] = pixBP[j] - BPref[n]

            if pixBP[j] < 0:            #Set negative values to NaN
                pixBP[j] = float("NaN")
           
            pixCNRP[j] = np.divide(pixCNRtarg[j],(pixCNRtarg[j] + pixCNRref[j]))
            if np.size(np.array(CNRPref))>1:#If CNRP reference matrix is present, determine error
                pixCNRPerror[j] = pixCNRP[j] - CNRPref[n]
            if pixCNRP[j] < 0:            #Set negative values to NaN
                pixCNRP[j] = float("NaN")

        #Increment counter for next object
        print(n)
        n = n+1
        if n == objectCount:
            n = 0

        #calculate mean and stdev for each object
        targCNRmean[i] = np.nanmean(pixCNRtarg[i])
        targCNRstd[i] = np.nanstd(pixCNRtarg[i])
        refCNRmean[i] = np.nanmean(pixCNRref[i])
        refCNRstd[i] = np.nanstd(pixCNRref[i])
        
        BPRMSE[i] = np.sqrt(np.nanmean(pixBPerror**2))
        CNRPRMSE[i] = np.sqrt(np.nanmean(pixCNRPerror**2))
        BP[i] = np.nanmean(pixBP)
        BPdev[i] = np.nanstd(pixBP)
        CNRP[i] = np.nanmean(pixCNRP)
        CNRPdev[i] = np.nanstd(pixCNRP)

    # Using DataFrame.assign() to add columns for each data point
    objectSignals = objectData.assign(
                   TargMean = targMean,
                   TargStdev = targStdev,
                   RefMean = refMean,
                   RefStdev = refStdev,
                   BackTargMean = backTargMean,
                   BackTargStdev = backTargStdev,
                   BackRefStdev = backRefStdev,
                   BackRefMean = backRefMean,
                   TargCNR = targCNR,
                   RefCNR = refCNR,
                   BP = BP,
                   CNRP = CNRP,
                   BPdev = BPdev,
                   CNRPdev = CNRPdev,
                   TargCNRmean = targCNRmean,
                   TargCNRstd = targCNRstd,
                   RefCNRmean = refCNRmean,
                   RefCNRstd = refCNRstd,
                   BPRMSE = BPRMSE,
                   CNRPRMSE = CNRPRMSE  
                   )

    return objectSignals

def createMaps(modelSignal, modelBackground, objectLabels, targ_chan, ref_chan, depthcorr):
    #Calculate average background and signal, and stdev for each pixel
    #initialize data structures
    targCNRmap = np.zeros(objectLabels.shape)
    refCNRmap = np.zeros(objectLabels.shape)
    BPmap = np.zeros(objectLabels.shape)
    CNRPmap = np.zeros(objectLabels.shape)

    #Create CNR maps
    for i in range(0,objectLabels.shape[2]):
        backgroundMean = np.nanmean(modelBackground[:,:,ref_chan,i])
        backgroundStdev = np.nanstd(modelBackground[:,:,ref_chan,i])
        if len(depthcorr)>1:#If depth correction matrix is present, apply it
            targCNRmap[:,:,i] = (modelSignal[:,:,targ_chan,i] - backgroundMean)/backgroundStdev/depthcorr[i]
        else:
            targCNRmap[:,:,i] = (modelSignal[:,:,targ_chan,i] - backgroundMean)/backgroundStdev
        refCNRmap[:,:,i] = (modelSignal[:,:,ref_chan,i] - backgroundMean)/backgroundStdev
        BPmap[:,:,i] = ndimage.median_filter(np.divide((targCNRmap[:,:,i] - refCNRmap[:,:,i]),refCNRmap[:,:,i]),size=3)
        CNRPmap[:,:,i] = ndimage.median_filter(np.divide(targCNRmap[:,:,i],(targCNRmap[:,:,i] + refCNRmap[:,:,i])),size=3)

    #Create BP maps, Create BP maps, with median2d filter in 3x3 area
    #Create CNRP maps, Create BP maps, with median2d filter
    return targCNRmap, refCNRmap, BPmap, CNRPmap

