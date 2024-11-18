static void quantize_all(DCAEncContext *c)

{

    int sample, band, ch;



    for (sample = 0; sample < SUBBAND_SAMPLES; sample++)

        for (band = 0; band < 32; band++)

            for (ch = 0; ch < c->fullband_channels; ch++)

                c->quantized[sample][band][ch] = quantize_value(c->subband[sample][band][ch], c->quant[band][ch]);

}
