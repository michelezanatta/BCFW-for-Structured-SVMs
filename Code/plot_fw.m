function plot_fw(iter_batch, iter_gap, iter_uni, time_batch, time_gap, time_uni, gap_tt_batch, gap_tt_gap, ...
    gap_tt_uni, avg_loss_tt_batch, avg_loss_tt_gap, avg_loss_tt_uni)



figure
semilogy(time_batch, gap_tt_batch, 'LineWidth',2.5)
xlim([0 20000])
ylim([0 2])
hold on
semilogy(time_gap, gap_tt_gap, 'LineWidth',2.5)
semilogy(time_uni, gap_tt_uni, 'LineWidth',2.5)
legend('Batch', 'Gap', 'Uni')
title('Duality gap by CPU time')
xlabel('CPU time'); ylabel('Gap');
hold off

figure
semilogy(iter_batch, gap_tt_batch, 'LineWidth',2.5)
xlim([0 5000])
ylim([0 2])
hold on
semilogy(iter_gap, gap_tt_gap, 'LineWidth',2.5)
semilogy(iter_uni, gap_tt_uni, 'LineWidth',2.5)
legend('Batch', 'Gap', 'Uni')
title('Duality gap by iterations')
xlabel('Iteration'); ylabel('Gap');
hold off




figure
semilogy(time_batch, avg_loss_tt_batch, 'LineWidth',2.5)
xlim([0 20000])
ylim([0 2])
hold on
semilogy(time_gap, avg_loss_tt_gap, 'LineWidth',2.5)
semilogy(time_uni, avg_loss_tt_uni, 'LineWidth',2.5)
legend('Batch', 'Gap', 'Uni')
title('Error on Training Set by CPU time')
xlabel('CPU time'); ylabel('Training Error');
hold off

avg_loss_tt_batch((length(avg_loss_tt_batch)+1):iter_uni(end)) = avg_loss_tt_batch(end);
iter_batch((length(iter_batch)+1):iter_uni(end)) = (length(iter_batch)+1):iter_uni(end);

figure
semilogy(iter_batch, avg_loss_tt_batch, 'LineWidth',2.5)
xlim([0 5000])
ylim([0 1])
hold on
semilogy(iter_gap, avg_loss_tt_gap, 'LineWidth',2.5)
semilogy(iter_uni, avg_loss_tt_uni, 'LineWidth',2.5)
legend('Batch', 'Gap', 'Uni')
title('Error on Training Set by iterations')
xlabel('Iteration'); ylabel('Training Error');
hold off