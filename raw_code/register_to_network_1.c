static void register_to_network(RDMARegister *reg)

{

    reg->key.current_addr = htonll(reg->key.current_addr);

    reg->current_index = htonl(reg->current_index);

    reg->chunks = htonll(reg->chunks);

}
