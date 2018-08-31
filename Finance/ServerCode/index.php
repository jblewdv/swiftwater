</!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Testing</title>
</head>
<body>
	<?php
		$servername = "localhost";
		$username = "root";
		$password = "strongpassword";
		$dbname = "stocklabels";
		// Create connection
		$conn = new mysqli($servername, $username, $password, $dbname);
		// Check connection
		if ($conn->connect_error) {
		    die("Connection failed: " . $conn->connect_error);
		}

		$sql = "SELECT table_name FROM information_schema.tables WHERE table_schema=$dbname";
		$result = $conn->query($sql);

		if ($result->num_rows > 0) {
			while($row = $result->fetch_assoc()) {
				$stock = $row['TABLE_NAME'];

				$sql2 = "SELECT * FROM " + $stock + " WHERE id='2018-08-14'";
				$newResults = $conn->query($sql2);

				while($row = $newResults->fetch_assoc()) {
					echo $row["Open"]. "<br>";
				//echo "Stock: " . $row["TABLE_NAME"]. "<br>";
				}
			}
		    // output data of each row
		    // while($row = $result->fetch_assoc()) {
		    //     echo "id: " . $row["id"]. "<br>";
		    // }
		} else {
		    echo "0 results";
		}
		$conn->close();
	?>

</body>
</html>