static int jpeg2000_read_bitstream_packets(Jpeg2000DecoderContext *s)

{

    int ret = 0;

    int tileno;



    for (tileno = 0; tileno < s->numXtiles * s->numYtiles; tileno++) {

        Jpeg2000Tile *tile = s->tile + tileno;



        if (ret = init_tile(s, tileno))

            return ret;



        s->g = tile->tile_part[0].tpg;

        if (ret = jpeg2000_decode_packets(s, tile))

            return ret;

    }



    return 0;

}
