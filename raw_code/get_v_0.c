static uint64_t get_v(ByteIOContext *bc)

{

    uint64_t val = 0;



    for(; bytes_left(bc) > 0; )

    {

	int tmp = get_byte(bc);



	if (tmp&0x80)

	    val= (val<<7) + tmp - 0x80;

	else

	    return (val<<7) + tmp;

    }

    return -1;

}
