static QUANT_FN(pvq_encode_band)

{

    return quant_band_template(pvq, f, rc, band, X, Y, N, b, blocks, lowband, duration,

                               lowband_out, level, gain, lowband_scratch, fill, 1);

}
