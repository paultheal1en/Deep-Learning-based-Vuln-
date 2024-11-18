int av_cold ff_mlp_init_crc2D(AVCodecParserContext *s)

{

    if (!crc_init_2D) {

        av_crc_init(crc_2D, 0, 16, 0x002D, sizeof(crc_2D));

        crc_init_2D = 1;

    }



    return 0;

}
