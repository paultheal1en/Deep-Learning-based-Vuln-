static void virtio_9p_get_config(VirtIODevice *vdev, uint8_t *config)

{

    struct virtio_9p_config *cfg;

    V9fsState *s = to_virtio_9p(vdev);



    cfg = g_malloc0(sizeof(struct virtio_9p_config) +

                        s->tag_len);

    stw_raw(&cfg->tag_len, s->tag_len);

    memcpy(cfg->tag, s->tag, s->tag_len);

    memcpy(config, cfg, s->config_size);

    g_free(cfg);

}
