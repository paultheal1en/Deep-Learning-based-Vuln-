static int get_packetheader(NUTContext *nut, ByteIOContext *bc, int calculate_checksum)

{

    int64_t start, size;

//    start= url_ftell(bc) - 8;



    size= get_v(bc);



    init_checksum(bc, calculate_checksum ? av_crc04C11DB7_update : NULL, 1);



//    nut->packet_start[2] = start;

//    nut->written_packet_size= size;



    return size;

}
