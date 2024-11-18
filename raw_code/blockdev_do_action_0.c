static void blockdev_do_action(int kind, void *data, Error **errp)

{

    TransactionAction action;

    TransactionActionList list;



    action.kind = kind;

    action.data = data;

    list.value = &action;

    list.next = NULL;

    qmp_transaction(&list, errp);

}
