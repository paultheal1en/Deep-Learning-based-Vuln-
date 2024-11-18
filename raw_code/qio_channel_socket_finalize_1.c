static void qio_channel_socket_finalize(Object *obj)
{
    QIOChannelSocket *ioc = QIO_CHANNEL_SOCKET(obj);
    if (ioc->fd != -1) {
#ifdef WIN32
        WSAEventSelect(ioc->fd, NULL, 0);
#endif
        closesocket(ioc->fd);
        ioc->fd = -1;