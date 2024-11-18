static void ehci_advance_state(EHCIState *ehci,

                               int async)

{

    EHCIQueue *q = NULL;

    int again;

    int iter = 0;



    do {

        if (ehci_get_state(ehci, async) == EST_FETCHQH) {

            iter++;

            /* if we are roaming a lot of QH without executing a qTD

             * something is wrong with the linked list. TO-DO: why is

             * this hack needed?

             */

            assert(iter < MAX_ITERATIONS);

#if 0

            if (iter > MAX_ITERATIONS) {

                DPRINTF("\n*** advance_state: bailing on MAX ITERATIONS***\n");

                ehci_set_state(ehci, async, EST_ACTIVE);

                break;

            }

#endif

        }

        switch(ehci_get_state(ehci, async)) {

        case EST_WAITLISTHEAD:

            again = ehci_state_waitlisthead(ehci, async);

            break;



        case EST_FETCHENTRY:

            again = ehci_state_fetchentry(ehci, async);

            break;



        case EST_FETCHQH:

            q = ehci_state_fetchqh(ehci, async);

            again = q ? 1 : 0;

            break;



        case EST_FETCHITD:

            again = ehci_state_fetchitd(ehci, async);

            break;



        case EST_FETCHSITD:

            again = ehci_state_fetchsitd(ehci, async);

            break;



        case EST_ADVANCEQUEUE:

            again = ehci_state_advqueue(q, async);

            break;



        case EST_FETCHQTD:

            again = ehci_state_fetchqtd(q, async);

            break;



        case EST_HORIZONTALQH:

            again = ehci_state_horizqh(q, async);

            break;



        case EST_EXECUTE:

            iter = 0;

            again = ehci_state_execute(q, async);

            break;



        case EST_EXECUTING:

            assert(q != NULL);

            again = ehci_state_executing(q, async);

            break;



        case EST_WRITEBACK:

            assert(q != NULL);

            again = ehci_state_writeback(q, async);

            break;



        default:

            fprintf(stderr, "Bad state!\n");

            again = -1;

            assert(0);

            break;

        }



        if (again < 0) {

            fprintf(stderr, "processing error - resetting ehci HC\n");

            ehci_reset(ehci);

            again = 0;

            assert(0);

        }

    }

    while (again);



    ehci_commit_interrupt(ehci);

}
