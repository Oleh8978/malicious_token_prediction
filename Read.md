Malicious Token Detection on Base

Predict malicious tokens on Base based on on-chain activities Overview

With the rapid growth of blockchain ecosystems such as Base, an Ethereum Layer 2 solution, token creation has become an increasingly common activity. However, this proliferation has led to the emergence of malicious tokens designed to perform fraudulent activities, such as stealing funds, facilitating scams, or exploiting vulnerabilities in smart contracts. Malicious tokens are cryptocurrencies or digital assets created with hidden or harmful intentions to deceive users, investors, or participants in the blockchain network. Identifying these tokens early is critical to protecting users and maintaining the integrity of the blockchain ecosystem. This problem is inherently challenging due to the pseudonymous nature of blockchain transactions and the high prevalence of obfuscation tactics employed by attackers. Intuitively, malicious tokens often exhibit patterns of suspicious behavior, such as abnormal transaction flows, centralized ownership, contract code vulnerabilities, or unusual tokenomic structures. Leveraging these characteristics through data-driven approaches is key to tackling the problem effectively.

To address this problem, a machine learning-based malicious token prediction system can be developed using on-chain transactions and token metadata. At a high level, this solution involves collecting and preprocessing on-chain data to extract relevant features indicative of malicious activity. Examples include ownership distribution and transaction activity patterns. These features are then fed into a predictive model, such as a Random Forest, Gradient Boosting, or Graph Neural Network (GNN), which is trained to classify tokens as suspicious or benign. The model can additionally be augmented with anomaly detection to handle unseen attack patterns, providing a robust method for protecting the network and its users.

If you are unfamiliar with the basic concepts in Crypto such as tokens and wallets, please start with our blog post "Blockchain 101". Otherwise, let's dive in!

Objective

This problem can be formulated as a supervised, binary classification task: Train a machine learning model on known malicious tokens to learn patterns and features that distinguish malicious tokens from legitimate ones.

Model Output
For a given token, assign it to one of two classes:

1: Malicious Token

0: Legitimate Token

Data
Data for this competition can be downloaded at the Datasets tab. A brief description of the data is given below. Please see Datasets for column-level details.

Malicious and legitimate tokens

Transactions involving the tokens

Transfers of the tokens

DEX swaps of the tokens (if any)

Malicious and legitimate tokens

A list of malicious token addresses and legitimate token addresses are provided in the training_tokens table. The legitimate tokens are a random sample from all tokens that are not malicious. The table contains token addresses and their labels (0=legitimate, 1=malicious). It also contains metadata of the tokens including their creation times, names, and symbols (if available). 

Transactions

Transactions with the token contracts in the training token set are provided in the transactions table. Each transaction has a unique identifier (TX_HASH), the address initiating the transaction (FROM_ADDRESS), the address being interacted with (TO_ADDRESS), the amount transacted (VALUE), and other related information. 

Transfers of the tokens

For tokens in the training set, their transfers are provided in the token_transfers table (for ERC-20 tokens) and nft_transfers table (for ERC-721 and ERC-1155 tokens). Each transfer inherits data such as block_timestamp and tx_hash from the associated transaction, but also contains parsed data including

Sending address of the transfer (From_Address or NFT_From_Address) which is not necessarily the same as the From Address of the transaction

Receiving address of the transfer (To_Address or NFT_To_Address)

Decimal-adjusted amount of the asset (Amount_Precise) and its USD value (Amount_USD). The USD value is not always available.

Address of the token being transferred (Contract_Address) or the NFT being transferred (NFT_Address)

The token ID if it is a NFT transfer (TokenID)

image (8).png
DEX swaps of the tokens

Not every token was traded on decentralized exchanges (DEX). For any token that was, swaps of the token on DEXs are provided in the dex_swaps table. Each swap inherits data such as block_timestamp and tx_hash from the associated transaction, but also contains parsed data including

The address of the token sent for swap (Token_In)

The address of the token being swapped to (Token_Out)

Amount of input token (Amount_In) and its USD value (Amount_In_USD)

Amount of token received (Amount_Out) and its USD value (Amount_Out_USD)

The address that initiate the swap (Origin_From_Address)

The address that receives the swapped token (TX_TO)

Evaluation

A test set of token addresses is provided in the test_tokens table. For each token address in the test set, please classify it into one of two classes: 0 (legitimate) or 1 (malicious). The predicted labels will be compared with the ground truth labels we have. The following metric will be assessed.

Accuracy: The overall percentage of correctly classified addresses. If your predicted label matches the true label, you score a point! The mathematical formula is:

image.png
Submission File

Once your model is ready, submit your predictions for the test addresses in a simple CSV file with two columns (The column names have to match below exactly or the evaluation will error out):

ADDRESS: Token addresses from the test set.

PRED: Your predicted labels (0 or 1).

Make sure to submit predictions for every address in the test set, as any missing predictions will be counted as incorrect. Here is an example submission file.

ADDRESS,PRED
0x8c68ec7995b29c6b33006e91e5993ba3fe5a1635,1
0xc47aec3468fe6ac9b5cf169e6d4f5f39def92220,0
0xe4d6c505b202385b214e9d050403d8da4b60ec02,1
0x978e91547652d2ad3a63cf9e7873fedfc73bf66f,0