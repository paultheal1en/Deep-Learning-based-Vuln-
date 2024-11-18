static void cryptodev_builtin_cleanup(

             CryptoDevBackend *backend,

             Error **errp)

{

    CryptoDevBackendBuiltin *builtin =

                      CRYPTODEV_BACKEND_BUILTIN(backend);

    size_t i;

    int queues = backend->conf.peers.queues;

    CryptoDevBackendClient *cc;



    for (i = 0; i < MAX_NUM_SESSIONS; i++) {

        if (builtin->sessions[i] != NULL) {

            cryptodev_builtin_sym_close_session(

                    backend, i, 0, errp);

        }

    }



    assert(queues == 1);



    for (i = 0; i < queues; i++) {

        cc = backend->conf.peers.ccs[i];

        if (cc) {

            cryptodev_backend_free_client(cc);

            backend->conf.peers.ccs[i] = NULL;

        }

    }



    cryptodev_backend_set_ready(backend, false);

}
