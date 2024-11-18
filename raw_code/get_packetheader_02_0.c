static int get_packetheader(NUTContext *nut, ByteIOContext *bc, int calculate_checksum)

{

    int64_t start, size;

    start= url_ftell(bc) - 8;



    init_checksum(bc, calculate_checksum ? update_adler32 : NULL, 0);



    size= get_v(bc);



    nut->packet_start[2] = start;

    nut->written_packet_size= size;



    return size;

}
