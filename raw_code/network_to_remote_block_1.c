static void network_to_remote_block(RDMARemoteBlock *rb)

{

    rb->remote_host_addr = ntohll(rb->remote_host_addr);

    rb->offset = ntohll(rb->offset);

    rb->length = ntohll(rb->length);

    rb->remote_rkey = ntohl(rb->remote_rkey);

}
