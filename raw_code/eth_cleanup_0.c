static void eth_cleanup(NetClientState *nc)

{

    ETRAXFSEthState *eth = qemu_get_nic_opaque(nc);



    /* Disconnect the client.  */

    eth->dma_out->client.push = NULL;

    eth->dma_out->client.opaque = NULL;

    eth->dma_in->client.opaque = NULL;

    eth->dma_in->client.pull = NULL;

        g_free(eth);

}
