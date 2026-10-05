J/ApJS/80/531  An X-ray catalog and atlas of galaxies.       (Fabbiano+, 1992)
================================================================================
An X-ray catalog and atlas of galaxies.
       Fabbiano G., Kim D.-W., Trinchieri G.
      <Astrophys. J. Suppl. Ser. 80, 531 (1992)>
      =1992ApJS...80..531F
================================================================================
ADC_Keywords: X-ray sources ; Galaxy catalogs
Mission_Name: Einstein
Keywords: atlases - catalogs - Galaxy: structure - X-rays: galaxies

Abstract:

    An  X-ray catalog  and atlas  of galaxies  observed with  the Einstein
    Observatory  imaging instruments  (IPC and  HRI)  are  presented.  The
    catalog  comprises   493  galaxies,   including   targets  of  pointed
    observations,  and RSA  or  RC2  galaxies serendipitously  included in
    Einstein fields.  A  total of 450  of these galaxies  were imaged well
    within  the instrumental fields,  resulting in 238 detections and 2123
    sigma upper limits.  The other galaxies were either at the edge of the
    visible field of view or confused with other X-ray sources.  For these
    a rough measure of their X-ray emission is also given. The atlas shows
    X-ray   contour  maps  of  detected  galaxies  superposed  on  optical
    photographs and gives azimuthally averaged surface brightness profiles
    of galaxies detected with a high signal-to-noise ratio.

File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl  Records  Explanations
--------------------------------------------------------------------------------
ReadMe         80         .  This file
gxfluxes.dat  123       448  Flux Density Info for the EINSTEIN Galaxy Catalog
--------------------------------------------------------------------------------

Byte-by-byte Description of file: gxfluxes.dat
--------------------------------------------------------------------------------
  Bytes Format  Units    Label    Explanations
--------------------------------------------------------------------------------
   2-  7 A6    ---       Name     Name of the object (NGC, IC, UGC number)
   9- 10 A2    ---       note     [ a-e] Flag indicating ending note (1)
  12- 13 I2    h         RAh      Right Ascension (B1950) (hour)
  15- 16 I2    min       RAm      Right Ascension (B1950) (min)
  18- 19 I2    s         RAs      Right Ascension (B1950) (sec)
      21 A1    ---       DE-      Declination sign (B1950)
  22- 23 I2    deg       DEd      Declination (B1950) (deg)
  25- 26 I2    arcmin    DEm      Declination (B1950) (arcmin)
  28- 29 I2    arcsec    DEs      Declination (B1950) (arcsec)
      31 A1    ---       instr    [IH] Instrument used (2)
  33- 37 I5    ---       AA    Observation sequence number (3)
  39- 43 F5.2  mag       Bmag     Apparent B magnitude (4)
  45- 46 I2    ---       type     T-type following RC2 (5)
  48- 49 I2    h         RA_Xh    ? Right Ascension (B1950) (hour) (6)
  51- 52 I2    min       RA_Xm    ? Right Ascension (B1950) (min)
  54- 55 I2    s         RA_Xs    ? Right Ascension (B1950) (sec)
      57 A1    ---       DE_X-    ? Declination sign (B1950) (6)
  58- 59 I2    deg       DE_Xd    ? Declination (B1950) (deg)
  61- 62 I2    arcmin    DE_Xm    ? Declination (B1950) (arcmin)
  64- 65 I2    arcsec    DE_Xs    ? Declination (B1950) (arcsec)
  67- 70 I4    arcsec    radius   ? Size of aperture for flux extraction
      72 A1    ---       l_counts Flagged '<' if counts is an upper limit (7)
  74- 81 F8.2  ct        counts   ? Total counts from EINSTEIN observation (7)
  83- 87 F5.1  ct        e_counts ? Error on count total (7)
      89 A1    ---       l_crate  [ <] '<' if count rate is upper limit (8)
  91- 97 F7.2  ct/ks     crate    ? Count-rate from EINSTEIN observation (8)
      99 A1    ---       l_FluxX  [ <] '<' if flux is an upper limit (9)
 101-107 F7.2  10-16W/m2 FluxX    ? X-ray flux (9)
 109-115 F7.3  Mpc       dist     Assumed distance in Megaparsecs (10)
     117 A1    ---       l_logLx  [ <] '<' if luminosity is upper limit (11)
 119-123 F5.2  [10-7W]   logLx    ? Log of the X-ray luminosity (11)
--------------------------------------------------------------------------------

Note (1):  This column gives a flag if there is a note on the source. The
          following are used:
             `a' if the galaxy has an AGN
             `b' if it is a Virgo cluster member
             `c' X-ray data are from the following:
                 LMC and SMC:  Long and van Speybroeck (1983) in "Accretion
                       Driven X-Ray Sources" Lewin and van den Heuvel eds.
                       (Cambridge:  Cambridge U. Press) p. 117.
                M31 and M32:  Trinchieri and Fabbiano (1991ApJ...382...82T)
                M33:  Trinchieri et al. (1988ApJ...325..531T)
                M101:  Trinchieri et al. (1990ApJ...356..110T)
         `d' no morphological type available (peculiar galaxy)
         `e' combined flux of two sources (see table 7 of published paper)

Note (2):  I for IPC, and H for HRI

Note (3): A sequence number of 99999 indicates that there is more than one
          observation of the source. See the gxatlas.dat table for
          details on the observations.

Note (4): This column gives the B-band apparent magnitude from the RSA, RC2
          or PCG (Patural's Principal Catalog of Galaxies). (Cat. <VII/119>)

Note (5): This column gives the T-type, as defined in the RC2 (Cat. <VII/112>),
          using morphological information from the RSA, RC2, or PCG.

Note (6): These columns give the B1950 right ascension and declination of
          the X-ray centroids of those objects detected by EINSTEIN.

Note (7): These columns give the total (background-subtracted) X-ray counts
          or upper limits for all galaxies in the catalog. The error in the
          counts (1-sigma) is also given for all detected galaxies. A "<" in
          the l_counts column indicates an upper limit. No error is given for
          upper limits.

Note (8): The crate column gives the EINSTEIN X-ray count-rate per
          1000 seconds. The l_crate column is flagged with a "<" if the
          count-rate is an upper limit.

Note (9): The FluxX column gives the EINSTEIN X-ray flux in units of
          10E-13 erg/sec/cm2. The l_FluxX column is flagged with a "<"
          if FluxX is an upper limit.

Note (10): This column gives the assumed distance in Mpc for all galaxies in
          the catalog, derived assuming a Hubble constant of 50 km/s/Mpc
          corrected for Virgocentric inflow.

Note (11): The logLx column gives the log of the derived X-ray luminosity
          (or upper-limit thereof) for all galaxies in the catalog, using
          the distances given in the distance column. Units are
          log(ergs/sec). The l_logLx column is flagged with a "<" if
          logLx is an upper limit.
--------------------------------------------------------------------------------

History:
    Prepared from the tables available at the "ADS Catalogue Service"
    (CfA, Harvard-Smithsonian Center for Astrophysics, Cambrigde MA)

References:
  Long and van Speybroeck (1983) in "Accretion Driven X-Ray Sources",
    Lewin and van den Heuvel eds.  (Cambridge:  Cambridge U. Press) p. 117.

================================================================================
(End)                                          Patricio Ortiz [CDS]  06-Apr-1999