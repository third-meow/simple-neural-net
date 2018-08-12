#simpleNN

A simple python nerual network. For education or something

##Attempt 2

No documentation on attempt 2 yet



##Attempt 1

###notes for anyone trying to understand this

- each layers 'weight' list is a 2D list
  eg.

  [[x, y, z],
   [p, q, r],
   [etc]]

  where x is the weight between the first neron in the layer in question 
  to first neron in the layer _before_

  similarly r is the weight between the 2ed neron in the layer in
  question to the 3ed neron in the layer before



- each layers 'cost' list is a 1D list of each nerons cost
  Per neron cost (the whole network has another cost) is
  calculated by averaging all the weights leading _from_
  it to the _next_ layer. These weights are found within the
  next layer's weights list
