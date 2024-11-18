static int build_vlc(VLC *vlc, const uint8_t *bits_table, const uint8_t *val_table, 

                      int nb_codes)

{

    uint8_t huff_size[256];

    uint16_t huff_code[256];



    memset(huff_size, 0, sizeof(huff_size));

    build_huffman_codes(huff_size, huff_code, bits_table, val_table);

    

    return init_vlc(vlc, 9, nb_codes, huff_size, 1, 1, huff_code, 2, 2);

}
