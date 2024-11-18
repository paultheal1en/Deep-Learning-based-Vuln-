static void ccid_card_vscard_send_msg(PassthruState *s,

        VSCMsgType type, uint32_t reader_id,

        const uint8_t *payload, uint32_t length)

{

    VSCMsgHeader scr_msg_header;



    scr_msg_header.type = htonl(type);

    scr_msg_header.reader_id = htonl(reader_id);

    scr_msg_header.length = htonl(length);

    qemu_chr_fe_write(s->cs, (uint8_t *)&scr_msg_header, sizeof(VSCMsgHeader));

    qemu_chr_fe_write(s->cs, payload, length);

}
