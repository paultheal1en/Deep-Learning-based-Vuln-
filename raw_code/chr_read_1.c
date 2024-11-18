static void chr_read(void *opaque, const uint8_t *buf, int size)
{
    TestServer *s = opaque;
    CharDriverState *chr = s->chr;
    VhostUserMsg msg;
    uint8_t *p = (uint8_t *) &msg;
    int fd;
    if (size != VHOST_USER_HDR_SIZE) {
        g_test_message("Wrong message size received %d\n", size);
        return;
    g_mutex_lock(&s->data_mutex);
    memcpy(p, buf, VHOST_USER_HDR_SIZE);
    if (msg.size) {
        p += VHOST_USER_HDR_SIZE;
        size = qemu_chr_fe_read_all(chr, p, msg.size);
        if (size != msg.size) {
            g_test_message("Wrong message size received %d != %d\n",
                           size, msg.size);
            return;
    switch (msg.request) {
    case VHOST_USER_GET_FEATURES:
        /* send back features to qemu */
        msg.flags |= VHOST_USER_REPLY_MASK;
        msg.size = sizeof(m.payload.u64);
        msg.payload.u64 = 0x1ULL << VHOST_F_LOG_ALL |
            0x1ULL << VHOST_USER_F_PROTOCOL_FEATURES;
        p = (uint8_t *) &msg;
        qemu_chr_fe_write_all(chr, p, VHOST_USER_HDR_SIZE + msg.size);
        break;
    case VHOST_USER_SET_FEATURES:
	g_assert_cmpint(msg.payload.u64 & (0x1ULL << VHOST_USER_F_PROTOCOL_FEATURES),
			!=, 0ULL);
        break;
    case VHOST_USER_GET_PROTOCOL_FEATURES:
        /* send back features to qemu */
        msg.flags |= VHOST_USER_REPLY_MASK;
        msg.size = sizeof(m.payload.u64);
        msg.payload.u64 = 1 << VHOST_USER_PROTOCOL_F_LOG_SHMFD;
        p = (uint8_t *) &msg;
        qemu_chr_fe_write_all(chr, p, VHOST_USER_HDR_SIZE + msg.size);
        break;
    case VHOST_USER_GET_VRING_BASE:
        /* send back vring base to qemu */
        msg.flags |= VHOST_USER_REPLY_MASK;
        msg.size = sizeof(m.payload.state);
        msg.payload.state.num = 0;
        p = (uint8_t *) &msg;
        qemu_chr_fe_write_all(chr, p, VHOST_USER_HDR_SIZE + msg.size);
        assert(msg.payload.state.index < 2);
        s->rings &= ~(0x1ULL << msg.payload.state.index);
        break;
    case VHOST_USER_SET_MEM_TABLE:
        /* received the mem table */
        memcpy(&s->memory, &msg.payload.memory, sizeof(msg.payload.memory));
        s->fds_num = qemu_chr_fe_get_msgfds(chr, s->fds, G_N_ELEMENTS(s->fds));
        /* signal the test that it can continue */
        g_cond_signal(&s->data_cond);
        break;
    case VHOST_USER_SET_VRING_KICK:
    case VHOST_USER_SET_VRING_CALL:
        /* consume the fd */
        qemu_chr_fe_get_msgfds(chr, &fd, 1);
        /*
         * This is a non-blocking eventfd.
         * The receive function forces it to be blocking,
         * so revert it back to non-blocking.
         */
        qemu_set_nonblock(fd);
        break;
    case VHOST_USER_SET_LOG_BASE:
        if (s->log_fd != -1) {
            close(s->log_fd);
            s->log_fd = -1;
        qemu_chr_fe_get_msgfds(chr, &s->log_fd, 1);
        msg.flags |= VHOST_USER_REPLY_MASK;
        msg.size = 0;
        p = (uint8_t *) &msg;
        qemu_chr_fe_write_all(chr, p, VHOST_USER_HDR_SIZE);
        g_cond_signal(&s->data_cond);
        break;
    case VHOST_USER_SET_VRING_BASE:
        assert(msg.payload.state.index < 2);
        s->rings |= 0x1ULL << msg.payload.state.index;
        break;
    default:
        break;
    g_mutex_unlock(&s->data_mutex);