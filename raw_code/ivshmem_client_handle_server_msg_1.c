ivshmem_client_handle_server_msg(IvshmemClient *client)
{
    IvshmemClientPeer *peer;
    long peer_id;
    int ret, fd;
    ret = ivshmem_client_read_one_msg(client, &peer_id, &fd);
    if (ret < 0) {
    /* can return a peer or the local client */
    peer = ivshmem_client_search_peer(client, peer_id);
    /* delete peer */
    if (fd == -1) {
        if (peer == NULL || peer == &client->local) {
            IVSHMEM_CLIENT_DEBUG(client, "receive delete for invalid "
                                 "peer %ld\n", peer_id);
        IVSHMEM_CLIENT_DEBUG(client, "delete peer id = %ld\n", peer_id);
        ivshmem_client_free_peer(client, peer);
        return 0;
    /* new peer */
    if (peer == NULL) {
        peer = g_malloc0(sizeof(*peer));
        peer->id = peer_id;
        peer->vectors_count = 0;
        QTAILQ_INSERT_TAIL(&client->peer_list, peer, next);
        IVSHMEM_CLIENT_DEBUG(client, "new peer id = %ld\n", peer_id);
    /* new vector */
    IVSHMEM_CLIENT_DEBUG(client, "  new vector %d (fd=%d) for peer id %ld\n",
                         peer->vectors_count, fd, peer->id);
    peer->vectors[peer->vectors_count] = fd;
    peer->vectors_count++;
    return 0;