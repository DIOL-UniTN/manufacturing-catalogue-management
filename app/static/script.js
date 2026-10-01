$(document).ready(function () {
  let selectedItem = null;

  const saved = localStorage.getItem('savedUserInput');
    if (saved) {
        $('#userInput').val(saved);
        
    }

  // Funzione per mostrare i risultati
  function renderResults(items) {
    $('#results').empty();
    selectedItem = null;
    $('#okBtn').prop('disabled', true);

    items.forEach(item => {
      const $item = $(`
        <div class="border rounded p-2 mb-2 result-item" style="cursor: pointer;">
          <div class="d-flex justify-content-between align-items-start">
            <small class="text-muted">ID: ${item.id}</small>
            <span class="badge bg-info text-dark">Score: ${item.score}</span>
          </div>
          <div class="text-truncate small mt-1">
            <strong>${item.description_ita}</strong> —
            <em class="text-muted">${item.description_eng}</em>
          </div>
        </div>
      `);

      $item.on('click', function (e) {
        e.stopPropagation();
        const alreadySelected = $(this).hasClass('border-primary');

        $('.result-item').removeClass('border-primary');
        selectedItem = null;
        $('#okBtn').prop('disabled', true);

        if (!alreadySelected) {
          $(this).addClass('border-primary');
          selectedItem = item;
          $('#okBtn').prop('disabled', false);
        }
      });

      $('#results').append($item);
    });
  }

  // Invio input con jQuery AJAX
  $('#submitBtn').on('click', function () {
    const input = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', input);
    if (input) {

      // clear previous results
      $('#results').empty();

      // show loading spinner
      $('#loadingSpinner').css('display', 'block');

      $.ajax({
        url: '/get_results',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ input: input }),
        success: function (data) {
          renderResults(data);
          console.log(data);
        },
        error: function () {
          alert('Errore nella richiesta al server.');
        },
        complete: function() {
          // hide loading spinner
          $('#loadingSpinner').css('display', 'none');
        }
      });
    }
  });

  $(document).on('keydown', function(event) {
    if (event.key === 'Enter') {
      const input = $('#userInput').val().trim();
      localStorage.setItem('savedUserInput', input);
      if (input) {
        // clear previous results
        $('#results').empty();

        // show loading spinner
        $('#loadingSpinner').show();

        $.ajax({
          url: '/get_results',
          method: 'POST',
          contentType: 'application/json',
          data: JSON.stringify({ input: input }),
          success: function (data) {
            renderResults(data);
          },
          error: function () {
            alert('Errore nella richiesta al server.');
          },
          complete: function() {
            // hide loading spinner
            $('#loadingSpinner').hide();
          }
        });
      }
    }
  });

  // Deseleziona tutto cliccando fuori
  $(document).on('click', function () {
    $('.result-item').removeClass('active');
    selectedItem = null;
    $('#okBtn').prop('disabled', true);
  });

  // Click su OK
  $('#okBtn').on('click', function () {
    if (selectedItem) {
      alert(`Hai selezionato: ${selectedItem}`);
    }
  });

  // Click su Create New
  $('#createNewBtn').on('click', function () {
      const input = $('#userInput').val().trim();
      localStorage.setItem('savedUserInput', input);
      window.location.href = '/insertion';
  });

    // Abilita/disabilita "Create New" in base all'input utente
  $('#userInput').on('input', function () {
    const isNotEmpty = $(this).val().trim().length > 0;
    $('#createNewBtn').prop('disabled', !isNotEmpty);
  });

  // Trigger iniziale per stato corretto su load
  $('#userInput').trigger('input');

});
