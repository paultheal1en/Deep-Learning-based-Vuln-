void pcmcia_socket_unregister(PCMCIASocket *socket)

{

    struct pcmcia_socket_entry_s *entry, **ptr;



    ptr = &pcmcia_sockets;

    for (entry = *ptr; entry; ptr = &entry->next, entry = *ptr)

        if (entry->socket == socket) {

            *ptr = entry->next;

            g_free(entry);

        }

}
