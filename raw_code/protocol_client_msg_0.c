static int protocol_client_msg(VncState *vs, char *data, size_t len)

{

    int i;

    uint16_t limit;



    switch (data[0]) {

    case 0:

	if (len == 1)

	    return 20;



	set_pixel_format(vs, read_u8(data, 4), read_u8(data, 5),

			 read_u8(data, 6), read_u8(data, 7),

			 read_u16(data, 8), read_u16(data, 10),

			 read_u16(data, 12), read_u8(data, 14),

			 read_u8(data, 15), read_u8(data, 16));

	break;

    case 2:

	if (len == 1)

	    return 4;



	if (len == 4)

	    return 4 + (read_u16(data, 2) * 4);



	limit = read_u16(data, 2);

	for (i = 0; i < limit; i++) {

	    int32_t val = read_s32(data, 4 + (i * 4));

	    memcpy(data + 4 + (i * 4), &val, sizeof(val));

	}



	set_encodings(vs, (int32_t *)(data + 4), limit);

	break;

    case 3:

	if (len == 1)

	    return 10;



	framebuffer_update_request(vs,

				   read_u8(data, 1), read_u16(data, 2), read_u16(data, 4),

				   read_u16(data, 6), read_u16(data, 8));

	break;

    case 4:

	if (len == 1)

	    return 8;



	key_event(vs, read_u8(data, 1), read_u32(data, 4));

	break;

    case 5:

	if (len == 1)

	    return 6;



	pointer_event(vs, read_u8(data, 1), read_u16(data, 2), read_u16(data, 4));

	break;

    case 6:

	if (len == 1)

	    return 8;



	if (len == 8)

	    return 8 + read_u32(data, 4);



	client_cut_text(vs, read_u32(data, 4), data + 8);

	break;

    default:

	printf("Msg: %d\n", data[0]);

	vnc_client_error(vs);

	break;

    }

	

    vnc_read_when(vs, protocol_client_msg, 1);

    return 0;

}
