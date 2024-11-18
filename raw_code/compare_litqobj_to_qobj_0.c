static int compare_litqobj_to_qobj(LiteralQObject *lhs, QObject *rhs)

{

    if (lhs->type != qobject_type(rhs)) {

        return 0;

    }



    switch (lhs->type) {

    case QTYPE_QINT:

        return lhs->value.qint == qint_get_int(qobject_to_qint(rhs));

    case QTYPE_QSTRING:

        return (strcmp(lhs->value.qstr, qstring_get_str(qobject_to_qstring(rhs))) == 0);

    case QTYPE_QDICT: {

        int i;



        for (i = 0; lhs->value.qdict[i].key; i++) {

            QObject *obj = qdict_get(qobject_to_qdict(rhs), lhs->value.qdict[i].key);



            if (!compare_litqobj_to_qobj(&lhs->value.qdict[i].value, obj)) {

                return 0;

            }

        }



        return 1;

    }

    case QTYPE_QLIST: {

        QListCompareHelper helper;



        helper.index = 0;

        helper.objs = lhs->value.qlist;

        helper.result = 1;

        

        qlist_iter(qobject_to_qlist(rhs), compare_helper, &helper);



        return helper.result;

    }

    default:

        break;

    }



    return 0;

}
