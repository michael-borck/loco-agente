# Queueing under heavy-tailed service times

## Abstract

Classical queueing theory assumes service times have light tails. Empirical
data from healthcare, customer support, and IT incident response often shows
heavy-tailed service times: most incidents resolve quickly but a long tail of
incidents takes orders of magnitude longer. We extend M/G/1 results to
heavy-tailed regimes and show that average wait times are dominated by the
tail rather than the body of the distribution. The practical implication for
operations: optimising for median service time can dramatically degrade
worst-case wait. Systems serving heavy-tailed loads need explicit slack capacity
or routing rules that protect the body from the tail.

## Caveats

The mathematical results assume stationarity, which empirical incident data
often violates. We discuss extensions to non-stationary regimes informally.
