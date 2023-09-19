from torchattacks import PGD, FGSM, PGDL2, AutoAttack, CW

# get attack 
def get_attack(net, args):
    if args.att_method == 'fgsm':
        attack = FGSM(net, args.epsilon)
        annotation = f'{args.att_method}_{args.att_norm}_eps({args.epsilon:.2f})'
    elif args.att_method == 'pgd':
        if args.att_norm == 'Linf':
            attack = PGD(net, eps=args.epsilon, alpha=args.alpha, steps=args.steps)
            annotation = f'{args.att_method}_{args.att_norm}_eps({args.epsilon:.2f})_alpha({args.alpha:.2f})_steps({args.steps})'

        elif args.att_norm == 'L2':
            attack = PGDL2(net, eps=args.epsilon, alpha=args.alpha, steps=args.steps)
            annotation = f'{args.att_method}_{args.att_norm}_eps({args.epsilon:.2f})_alpha({args.alpha:.2f})_steps({args.steps})'

    elif args.att_method == 'cw':
        attack = CW(net, c=args.c, kappa=args.kappa, steps=args.steps, lr=args.lr)
        annotation = f'{args.att_method}_{args.att_norm}_c({args.c})_kappa({args.kappa})_lr({args.lr})_steps({args.steps})'

    elif args.att_method == 'aa':
        attack = AutoAttack(net, norm=args.att_norm, eps=args.epsilon)
        annotation = f'{args.att_method}_{args.att_norm}_eps({args.epsilon:.2f})'

    return annotation, attack

